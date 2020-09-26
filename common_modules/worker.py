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

import common_modules.utils as utils
from common_modules.client_functions import send_exp_c, send_model_c, request_model, join_population, exit_population
from common_modules.env_utils import init_pos, lower_ver, get_space, all_zeros, get_normalized_state, get_counter, step
from common_modules.nets import Actor, Critic
from common_modules.noises import OUNoise, AdaptiveParamNoiseSpec, distance_metric
from common_modules.replay_buffer import ReplayBufferM
from common_modules.agent import Agent
from conf.default import Settings
from common_modules.utils import str2bool


def run_worker():
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
    parser.add_argument('--pull_pos', type=float, default=0.8)
    parser.add_argument('--gau_pos', type=float, default=0.2)
    parser.add_argument('--gau_mul', type=float, default=0.01)
    parser.add_argument('--client_id', type=int, default=2)
    parser.add_argument('--replay_start_size', type=int, default=1000)
    parser.add_argument('--max_episodes', type=int, default=100000)

    parser.add_argument('--replay_size', type=int)
    parser.add_argument('--updates_per_step', type=int)
    parser.add_argument('--test_episodes', type=int)
    parser.add_argument('--gen_period', type=int)

    parser.add_argument('--n_users', type=int)
    parser.add_argument('--n_channels', type=int)
    parser.add_argument('--alpha_p', type=float)
    parser.add_argument('--c_band', type=float)
    parser.add_argument('--noise', type=float)
    parser.add_argument('--power_tr', type=float)
    parser.add_argument('--backfoul', type=float)
    parser.add_argument('--cover_range', type=float)
    parser.add_argument('--max_episode_steps', type=int)

    parser.add_argument('--frequency_d', type=float, default=0.8)
    parser.add_argument('--frequency_de', type=float, default=5)
    parser.add_argument('--len_epoch', type=float, default=0.6)
    parser.add_argument('--Theta', type=float, default=1.)
    parser.add_argument('--unit_local', type=float, default=0.025)
    parser.add_argument('--unit_edge', type=float, default=0.05)
    parser.add_argument('--unit_j', type=float, default=0.1)
    parser.add_argument('--timestep_limit', type=int, default=50)
    parser.add_argument('--count', type=int, default=0)
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
    exp_pre_idx = 0
    model_types = ['a', 'at', 'ap', 'c', 'ct']

    join_population(args.client_id, Settings.master_host, Settings.master_port)
    print('start training client', args.client_id)
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
            obs_users[i_user][4] = random.uniform(0.6, 0.8)
            obs_users[i_user][5] = 2.0
            obs_users[i_user][6] = random.uniform(5000, 8000)

        obs_users[args.n_users][0] = args.frequency_de
        obs_users[args.n_users][1] = 0.0
        obs_users[args.n_users][2] = args.count

        state = torch.tensor([np.concatenate(get_normalized_state(obs_users, args.n_users), axis=0)]).float().to(
            device)

        episode_reward = 0

        for i_step in range(args.max_episode_steps):
            action = agent.select_action(state).to(device)

            np_action = action.cpu().numpy()[0]
            counter = get_counter(np_action, args.n_users)

            reward, done = step(obs_users, np_action, args.n_users, pos, args, counter)
            episode_reward += reward

            mask = torch.tensor([not done]).float().to(device)
            reward = torch.tensor([reward]).float().to(device)
            next_state = torch.tensor(
                [np.concatenate(get_normalized_state(obs_users, args.n_users), axis=0)]).float().to(device)

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

        # a more flexible approach to control genetation and experience update
        cur_cycle = int(agent.replay_buffer.total_count/agent.replay_buffer.capacity)
        if cur_cycle > exp_pre_idx:
            # represent a new cycle
            exp_pre_idx = cur_cycle
            print('save and upload exp for cycle', cur_cycle)
            exp_np = np.array(agent.replay_buffer.buffer)
            np.save('exps/exp_' + str(args.client_id) + '_' + str(cur_cycle) + '_' + str(agent.replay_buffer.capacity), exp_np)
            path = 'exps/exp_' + str(args.client_id) + '_' + str(cur_cycle) + '_' + str(agent.replay_buffer.capacity) + '.npy'
            send_exp_c(args.client_id, cur_cycle, path, Settings.master_host, Settings.master_port)

        writer.add_scalar('reward/train', episode_reward, i_episode)
        agent.update_param_noise()

        rewards.append(episode_reward)
        if i_episode % args.gen_period == 0 and i_episode > 0:
            # -1 for start with zero
            gen = int(i_episode/args.gen_period-1)

            if random.random() < args.gau_pos:
                print('apply noise for generation', gen)
                mut_actor_params = []
                for param in agent.actor.parameters():
                    temp = torch.randn(param.shape).to(device)
                    temp = temp * args.gau_mul
                    out = torch.add(param, temp)
                    mut_actor_params.append(out)

                for target_param, param in zip(agent.actor.parameters(), mut_actor_params):
                    target_param.data.copy_(param)

                mut_critic_params = []
                for param in agent.critic.parameters():
                    temp = torch.randn(param.shape).to(device)
                    out = torch.add(param, temp)
                    mut_critic_params.append(out)

                for target_param, param in zip(agent.critic.parameters(), mut_critic_params):
                    target_param.data.copy_(param)

            test_rewards = []
            for i_test in range(args.test_episodes):

                args.count = 1
                for i_user in range(args.n_users):
                    obs_users[i_user][0] = args.frequency_d
                    obs_users[i_user][1] = 0
                    obs_users[i_user][2] = pos[i_user * 2]
                    obs_users[i_user][3] = pos[i_user * 2 + 1]
                    obs_users[i_user][4] = random.uniform(0.6, 0.8)
                    obs_users[i_user][5] = 2.0
                    obs_users[i_user][6] = random.uniform(5000, 8000)

                obs_users[args.n_users][0] = args.frequency_de
                obs_users[args.n_users][1] = 0.0
                obs_users[args.n_users][2] = args.count

                state = torch.tensor(
                    [np.concatenate(get_normalized_state(obs_users, args.n_users), axis=0)]).float().to(
                    device)

                episode_reward = 0
                for i_step in range(args.max_episode_steps):
                    action = agent.select_action(state)

                    np_action = action.cpu().numpy()[0]
                    counter = get_counter(np_action, args.n_users)

                    reward, done = step(obs_users, np_action, args.n_users, pos, args, counter)
                    next_state = torch.tensor(
                        [np.concatenate(get_normalized_state(obs_users, args.n_users), axis=0)]).float().to(device)

                    episode_reward += reward

                    state = next_state
                    if done:
                        break

                # writer.add_scalar('reward/test', episode_reward, i_episode)
                test_rewards.append(episode_reward)

            fitness = sum(test_rewards)/len(test_rewards)
            print('client id:', str(args.client_id), 'i_episode', i_episode, 'fitness', fitness)

            actor_path, actor_target_path, actor_perturbed_path = \
                agent.save_actor_params(args.env_name, id=str(args.client_id), g=str(gen))
            critic_path, critic_target_path = \
                agent.save_critic_params(args.env_name, id=str(args.client_id), g=str(gen))

            paths = [actor_path, actor_target_path, actor_perturbed_path, critic_path, critic_target_path]

            for idx_c, path in enumerate(paths):
                send_model_c(path, fitness, args.client_id, gen, model_types[idx_c], Settings.master_host,
                             Settings.master_port)


            if random.random() < args.pull_pos:
                print('pull generation for cycle', gen)
                model_paths = request_model(args.client_id, fitness, Settings.master_host, Settings.master_port,
                                           str(args.client_id) + '_temp/')
                if len(model_paths) == 5:
                    agent.load_full_model(model_paths)

        # agent.save_model(args.env_name, suffix='0', id=str(args.client_id), g=str(gen))

    exit_population(args.client_id, Settings.master_host, Settings.master_port)

    # env.monitor.close()


if __name__ == '__main__':
    run_worker()

