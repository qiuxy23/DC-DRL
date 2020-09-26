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

import common_modules.utils as utils
from common_modules.nets import Actor, Critic
from common_modules.noises import OUNoise, AdaptiveParamNoiseSpec, distance_metric
from common_modules.replay_buffer import ReplayBufferM
from common_modules.agent import Agent
from conf.default import Settings
from common_modules.utils import str2bool

def run_evaluate():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='')
    parser.add_argument('--gpu', type=str2bool, default=True)
    parser.add_argument('--save', type=str2bool, default=False)
    parser.add_argument('--gamma', type=float, default=0.9, metavar='G')
    parser.add_argument('--tau', type=float, default=0.001, metavar='G')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N')
    parser.add_argument('--replay_size', type=int, default=100000, metavar='N')
    parser.add_argument('--replay_start_size', type=int, default=10000)
    parser.add_argument('--n_step', type=str2bool, default=True)
    parser.add_argument('--ou_noise', type=str2bool, default=True)
    parser.add_argument('--param_noise', type=str2bool, default=False)
    parser.add_argument('--updates_per_step', type=int, default=4)

    args = parser.parse_args()

    ENV_NAME = args.env_name
    EPISODES = 100000
    TEST = 50

    total_numsteps = 0

    env = gym.make(args.env_name)

    if args.gpu:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    writer = SummaryWriter()

    agent = Agent(env_name=ENV_NAME, obs_space=env.observation_space, act_space=env.action_space,
                  device=device, writer=writer, mode=None, gamma=args.gamma, batch_size=args.batch_size,
                  tau=args.tau, replay_size=args.replay_size, replay_start_size=args.replay_start_size,
                  n_step=args.n_step, ou_noise=args.ou_noise, param_noise=args.param_noise,
                  updates_per_step=args.updates_per_step)
    agent.cuda()

    # env.monitor.start('experiments/' + ENV_NAME, force=True)

    rewards = []
    for i_episode in range(EPISODES):
        if args.n_step:
            agent.reset_storage()

        state = torch.tensor([env.reset()]).float().to(device)
        episode_reward = 0

        for i_step in range(env.max_episode_steps):
            action = agent.select_action(state).to(device)
            next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
            total_numsteps += 1
            episode_reward += reward

            mask = torch.tensor([not done]).float().to(device)
            next_state = torch.tensor([next_state]).float().to(device)
            reward = torch.tensor([reward]).float().to(device)

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

        writer.add_scalar('reward/train', episode_reward, i_episode)
        agent.update_param_noise()

        rewards.append(episode_reward)
        if i_episode % TEST == 0:
            state = torch.tensor([env.reset()]).float().to(device)
            episode_reward = 0
            for i_step in range(env.max_episode_steps):
                action = agent.select_action(state)

                next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
                episode_reward += reward

                next_state = torch.tensor([next_state]).float().to(device)

                state = next_state
                if done:
                    break

            writer.add_scalar('reward/test', episode_reward, i_episode)

            rewards.append(episode_reward)
            print('Episode: {}, total numsteps: {}, reward: {}, average reward: '
                  '{}, time: {}'.format(i_episode, total_numsteps, rewards[-1],
                                        np.mean(rewards[-10:]), time.time()))

    if Settings.IS_GYM:
        env.monitor.close()

