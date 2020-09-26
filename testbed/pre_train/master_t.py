import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from tensorboardX import SummaryWriter
from torch.optim import Adam
import numpy as np
import time
import gym
import random
import shutil

import common_modules.utils as utils
from common_modules.client_functions import send_exp_c, send_model_c, request_model
from common_modules.nets import Actor, Critic
from common_modules.noises import OUNoise, AdaptiveParamNoiseSpec, distance_metric
from common_modules.replay_buffer import ReplayBufferM
from common_modules.agent import Agent
from conf.default import Settings
from common_modules.utils import str2bool
from common_modules.env_utils import init_pos, lower_ver, get_space, all_zeros, get_normalized_state_tb, get_counter, step, \
    step_tb


def run_master():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='')
    parser.add_argument('--gpu', type=str2bool, default=True)
    parser.add_argument('--save', type=str2bool, default=False)
    parser.add_argument('--gamma', type=float, default=0.9, metavar='G')
    parser.add_argument('--tau', type=float, default=0.001, metavar='G')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N')
    parser.add_argument('--n_step', type=str2bool, default=False)
    parser.add_argument('--ou_noise', type=str2bool, default=True)
    parser.add_argument('--param_noise', type=str2bool, default=False)
    parser.add_argument('--actor_load_path', type=str, default='')
    parser.add_argument('--critic_load_path', type=str, default='')
    parser.add_argument('--client_id', type=int, default=1)
    parser.add_argument('--pull_pos', type=float, default=0.8)
    parser.add_argument('--gau_pos', type=float, default=0.2)
    parser.add_argument('--gau_mul', type=float, default=0.01)
    parser.add_argument('--exp_path', type=str, default='recv/exps/')
    parser.add_argument('--client_replay_size', type=int, default=10000)
    parser.add_argument('--shift_range', type=int, default=1)
    parser.add_argument('--master_id', type=int, default=0)
    parser.add_argument('--load_pos', type=float, default=0.5)
    parser.add_argument('--replay_start_size', type=int, default=100) # we shorter this as the testbed env is much simplier than sim.
    parser.add_argument('--max_episodes', type=int, default=100000)
    parser.add_argument('--replay_size', type=int, default=10000)
    parser.add_argument('--test_episodes', type=int, default=1)  # as the video files are fixed

    parser.add_argument('--updates_per_step', type=int)
    parser.add_argument('--gen_period', type=int)

    parser.add_argument('--n_users', type=int)
    parser.add_argument('--n_channels', type=int)
    parser.add_argument('--alpha_p', type=float)
    parser.add_argument('--c_band', type=float)
    parser.add_argument('--noise', type=float)
    parser.add_argument('--power_tr', type=float)
    parser.add_argument('--backfoul', type=float)
    parser.add_argument('--cover_range', type=float)

    parser.add_argument('--frequency_d', type=float, default=1.2)  # Raspberry pi
    parser.add_argument('--frequency_de', type=float, default=2.8 * 8)
    parser.add_argument('--max_episode_steps', type=int, default=23)  # as there are 23 videos
    parser.add_argument('--len_epoch', type=float, default=360)
    parser.add_argument('--Theta', type=float, default=50)
    parser.add_argument('--unit_local', type=float, default=0.025)
    parser.add_argument('--unit_edge', type=float, default=0.05)
    parser.add_argument('--unit_j', type=float, default=0.1)
    parser.add_argument('--timestep_limit', type=int, default=50)
    parser.add_argument('--count', type=int, default=0)
    parser.add_argument('--ddl', type=int, default=600)
    parser.add_argument('--seed', type=int, default=2020)


    args = parser.parse_args()

    ENV_NAME = args.env_name

    total_numsteps = 0

    # basic information for env
    args.count = 0
    pos = init_pos(args.seed, args.n_users, args.cover_range)
    obs_users = []
    obs_len = 0
    for i_user in range(args.n_users):
        obs_users.append([0 for _ in range(7)])
        obs_len += 7
    obs_users.append([0, 0, 0])
    obs_len += 3

    # if Settings.IS_GYM:
    #     env = utils.makeFilteredEnv(gym.make(ENV_NAME))
    # else:

    observation_space, action_space = get_space(obs_len, args.n_users * 2)

    if args.gpu:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    writer = SummaryWriter()

    agent = Agent(env_name=ENV_NAME, obs_space=observation_space, act_space=action_space,
                  device=device, writer=writer, mode=None, gamma=args.gamma, batch_size=args.batch_size,
                  tau=args.tau, replay_size=args.replay_size, replay_start_size=args.replay_start_size,
                  n_step=args.n_step, ou_noise=args.ou_noise, param_noise=args.param_noise)

    agent.cuda()

    if args.actor_load_path != '' and args.critic_load_path != '':
        agent.load_model_with_hard_update(args.actor_load_path, args.critic_load_path)

    # env.monitor.start('experiments/' + ENV_NAME, force=True)

    rewards = []
    master_total_counter = 0
    model_types = ['a', 'at', 'ap', 'c', 'ct']

    sizes = np.load('../conf/sizes.npy')
    cycles = np.load('../conf/cycles.npy')

    for i_episode in range(args.max_episodes):
        # print('i_episode', i_episode)
        if args.n_step:
            agent.reset_storage()

        # obs_users = all_zeros(args.n_users)
        args.count = 1
        for i_user in range(args.n_users):
            obs_users[i_user][0] = args.frequency_d
            obs_users[i_user][1] = 0
            obs_users[i_user][2] = pos[i_user * 2]
            obs_users[i_user][3] = pos[i_user * 2 + 1]
            obs_users[i_user][4] = cycles[0]/400
            obs_users[i_user][5] = args.ddl
            obs_users[i_user][6] = sizes[0]/1024

        obs_users[args.n_users][0] = args.frequency_de
        obs_users[args.n_users][1] = 0.0
        obs_users[args.n_users][2] = args.count

        state = torch.tensor([np.concatenate(get_normalized_state_tb(obs_users, args.n_users), axis=0)]).float().to(
            device)

        episode_reward = 0

        for i_step in range(args.max_episode_steps):
            action = agent.select_action(state).to(device)

            np_action = action.cpu().numpy()[0]

            counter = get_counter(np_action, args.n_users)

            reward, done = step_tb(obs_users, np_action, args.n_users, pos, args, counter, cycles[i_step + 1], sizes[i_step +1])
            episode_reward += reward

            mask = torch.tensor([not done]).float().to(device)
            reward = torch.tensor([reward]).float().to(device)
            next_state = torch.tensor(
                [np.concatenate(get_normalized_state_tb(obs_users, args.n_users), axis=0)]).float().to(device)

            total_numsteps += 1

            if args.n_step:
                agent.adv_preceive(state, action, mask, next_state, reward)
            else:
                agent.preceive(state, action, mask, next_state, reward)

            if len(agent.replay_buffer) > agent.replay_start_size:
                for _ in range(agent.updates_per_step):
                    batch = agent.replay_buffer.sample(agent.batch_size)
                    # batch = Transition(*zip(*transitions))

                    if args.n_step:
                        value_loss, policy_loss = agent.adv_update(batch)
                    else:
                        value_loss, policy_loss = agent.update(batch)

                    if agent.writer:
                        agent.writer.add_scalar('loss/value', value_loss, agent.updates)
                        agent.writer.add_scalar('loss/policy', policy_loss, agent.updates)

                    agent.updates += 1

            state = next_state

            if done:
                agent.reset_noise(i_episode)
                break

        rewards.append(episode_reward)
        writer.add_scalar('reward/train', episode_reward, i_episode)
        agent.update_param_noise()

        if random.random() < args.load_pos:
            cycle = int(master_total_counter/args.client_replay_size)
            cycles_str = []
            for i_cycle in range(cycle - args.shift_range, cycle + args.shift_range + 1):
                cycles_str.append(str(i_cycle))
            selected_exp_paths, selected_dirs = utils.get_timely_exp_dirs(args.exp_path, cycles_str)

            for path, s_dir in zip(selected_exp_paths, selected_dirs):
                try:
                    selected_exps = np.load(path, allow_pickle=True)
                    print('load experience from', path)
                    for exp in selected_exps:
                        agent.preceive(exp[0], exp[1], exp[2], exp[3], exp[4])
                except:
                    print('broken files')

                shutil.rmtree(s_dir)

        if i_episode % args.gen_period == 0 and i_episode > 0:
            # -1 for start with zero
            gen = int(i_episode/args.gen_period-1)

            test_rewards = []
            for i_test in range(args.test_episodes):
                args.count = 1
                for i_user in range(args.n_users):
                    obs_users[i_user][0] = args.frequency_d
                    obs_users[i_user][1] = 0
                    obs_users[i_user][2] = pos[i_user * 2]
                    obs_users[i_user][3] = pos[i_user * 2 + 1]
                    obs_users[i_user][4] = cycles[0]/400
                    obs_users[i_user][5] = args.ddl
                    obs_users[i_user][6] = sizes[0]/400

                obs_users[args.n_users][0] = args.frequency_de
                obs_users[args.n_users][1] = 0.0
                obs_users[args.n_users][2] = args.count

                state = torch.tensor(
                    [np.concatenate(get_normalized_state_tb(obs_users, args.n_users), axis=0)]).float().to(
                    device)
                episode_reward = 0
                for i_step in range(args.max_episode_steps):
                    action = agent.select_action(state)

                    np_action = action.cpu().numpy()[0]
                    counter = get_counter(np_action, args.n_users)

                    reward, done = step_tb(obs_users, np_action, args.n_users, pos, args, counter, cycles[i_step + 1], sizes[i_step + 1])
                    next_state = torch.tensor(
                        [np.concatenate(get_normalized_state_tb(obs_users, args.n_users), axis=0)]).float().to(device)

                    episode_reward += reward

                    state = next_state
                    if done:
                        break

                # writer.add_scalar('reward/test', episode_reward, i_episode)
                test_rewards.append(episode_reward)

            fitness = sum(test_rewards)/len(test_rewards)

            print('master:', 'i_episode', i_episode, 'fitness', fitness)

            actor_path, actor_target_path, actor_perturbed_path = \
                agent.save_actor_params(args.env_name, g=str(gen))
            critic_path, critic_target_path = \
                agent.save_critic_params(args.env_name, g=str(gen))

            paths = [actor_path, actor_target_path, actor_perturbed_path, critic_path, critic_target_path]

            for idx_c, path in enumerate(paths):
                send_model_c(path, fitness, args.master_id, gen, model_types[idx_c], Settings.master_host,
                             Settings.master_port)

        # agent.save_model(args.env_name, suffix='0', id=str(args.master_id), g=str(gen))

    # env.monitor.close()



if __name__ == '__main__':
    run_master()
