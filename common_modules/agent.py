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
from common_modules.utils import str2bool

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class Agent(object):
    # agent class, a common module used in both master and distributed work
    def __init__(self,
                 env_name,
                 obs_space,
                 act_space,
                 device,
                 writer,
                 mode,
                 is_continue=True,
                 eval_env=None,
                 gamma=0.99,
                 tau=0.001,
                 replay_size=100000,
                 batch_size=64,
                 replay_start_size=10000,
                 n_step=False,
                 ou_noise=True,
                 param_noise=True,
                 noise_scale=0.3,
                 final_noise_scale=0.3,
                 exploration_end=100,
                 updates_per_step=1,
                 max_n=10,
                 critic_lr=1e-3,
                 actor_lr=1e-4):

        # env
        self.env_name = env_name
        self.obs_space = obs_space
        self.act_space = act_space
        self.obs_dim = self.obs_space.shape[0]
        self.act_dim = self.act_space.shape[0]

        # conf
        self.mode = mode
        self.device = device
        self.eval_env = eval_env
        self.is_continue = is_continue
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr

        # replay
        self.replay_start_size = replay_start_size
        self.n_step = n_step
        self.updates_per_step = updates_per_step

        # exp
        self.noise_scale = noise_scale
        self.final_noise_scale = final_noise_scale
        self.exploration_end = exploration_end

        # log
        self.writer = writer

        self.actor_net_dims = [400, 300]
        self.actor_net_archs = [F.relu, F.relu]
        self.actor = Actor(self.obs_space, self.act_space, self.actor_net_dims, self.actor_net_archs, 'tanh', None)
        self.actor_target = Actor(self.obs_space, self.act_space, self.actor_net_dims, self.actor_net_archs, 'tanh', None)
        self.actor_perturbed = Actor(self.obs_space, self.act_space, self.actor_net_dims, self.actor_net_archs, 'tanh', None)
        self.actor_optim = Adam(self.actor.parameters(), lr=self.actor_lr)


        self.critic_net_dims = [400, 300]
        self.critic_net_archs = [F.relu, F.relu]
        self.critic = Critic(self.obs_space, self.act_space, self.critic_net_dims, self.critic_net_archs, 'mul')
        self.critic_target = Critic(self.obs_space, self.act_space, self.critic_net_dims, self.critic_net_archs, 'mul')
        self.critic_optim = Adam(self.critic.parameters(), lr=self.critic_lr)

        self.gamma = gamma
        self.tau = tau

        # Make sure target is with the same weight
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

        self.replay_buffer = ReplayBufferM(replay_size)
        self.batch_size = batch_size

        self.ou_noise = OUNoise(self.act_space.shape[0]) if ou_noise else None
        self.param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.05,
                                             desired_action_stddev=noise_scale,
                                             adaptation_coefficient=1.05) if param_noise else None
        self.ou_noise.scale = self.noise_scale
        self.updates = 0


        self.episode_storage = []
        self.episode_storage_start = 0
        self.max_n = max_n
        self.episode_storage_end = max_n

    def cuda(self):
        self.actor.to(self.device)
        self.actor_target.to(self.device)
        self.actor_perturbed.to(self.device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)

    def select_action(self, state):
        self.actor.eval()
        if self.param_noise:
            mu = self.actor_perturbed((state))
        else:
            mu = self.actor((state))

        self.actor.train()
        mu = mu.data

        if self.ou_noise:
            mu += torch.tensor(self.ou_noise.noise()).float().to(self.device)

        return mu.clamp(-1, 1)

    def eval_action(self, state):
        self.actor.eval()
        mu = self.actor((state))

        self.actor.train()
        mu = mu.data

        return mu.clamp(-1, 1)


    def update(self, batch):
        state_batch = torch.cat([data[0] for data in batch])
        action_batch = torch.cat([data[1] for data in batch])
        mask_batch = torch.cat([data[2] for data in batch])
        next_state_batch = torch.cat([data[3] for data in batch])
        reward_batch = torch.cat([data[4] for data in batch])

        next_action_batch = self.actor_target(next_state_batch)
        next_state_action_values = self.critic_target(next_state_batch, next_action_batch)

        reward_batch = reward_batch.unsqueeze(1)
        mask_batch = mask_batch.unsqueeze(1)
        expected_state_action_batch = reward_batch + (self.gamma * mask_batch * next_state_action_values)

        self.critic_optim.zero_grad()

        state_action_batch = self.critic((state_batch), (action_batch))

        value_loss = F.mse_loss(state_action_batch, expected_state_action_batch)
        value_loss.backward()
        self.critic_optim.step()

        self.actor_optim.zero_grad()

        policy_loss = -self.critic((state_batch), self.actor((state_batch)))

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()

    def preceive(self, state, action, mask, next_state, reward):
        experience = (state, action, mask, next_state, reward)
        self.replay_buffer.push(experience)

    def adv_update(self, batch):

        state_batch = torch.cat([data[0][0] for data in batch])
        action_batch = torch.cat([data[0][1] for data in batch])
        next_state_batch, mask_batch = [], []
        reward_batch = []

        for data in batch:
            reward_temp = data[0][4]
            flag = False
            for i_data in range(1, len(data)):
                if self.follow_policy(data[i_data][0], data[i_data][1]):
                    reward_temp = reward_temp + pow(self.gamma, i_data) * data[i_data][4]
                else:
                    next_state_batch.append(data[i_data - 1][3])
                    mask_batch.append(data[i_data - 1][2])
                    flag = True
                    break

            if not flag:
                next_state_batch.append(data[len(data) - 1][3])
                mask_batch.append(data[len(data) - 1][2])
            reward_batch.append(reward_temp)

        next_state_batch = torch.cat(next_state_batch)
        mask_batch = torch.cat(mask_batch)
        reward_batch = torch.cat(reward_batch)

        # next_state_batch = torch.cat([data[0][1] for data in batch])
        # mask_batch = torch.cat([data[0][1] for data in batch])
        # reward_batch = torch.cat([data[0][1] for data in batch])

        # state_batch = torch.cat(batch.state)
        # action_batch = torch.cat(batch.action)
        # reward_batch = torch.cat(batch.reward)
        # mask_batch = torch.cat(batch.mask)
        # next_state_batch = torch.cat(batch.next_state)

        next_action_batch = self.actor_target(next_state_batch)
        next_state_action_values = self.critic_target(next_state_batch, next_action_batch)

        reward_batch = reward_batch.unsqueeze(1)
        mask_batch = mask_batch.unsqueeze(1)
        expected_state_action_batch = reward_batch + (self.gamma * mask_batch * next_state_action_values)

        self.critic_optim.zero_grad()

        state_action_batch = self.critic((state_batch), (action_batch))

        value_loss = F.mse_loss(state_action_batch, expected_state_action_batch)
        value_loss.backward()
        self.critic_optim.step()

        self.actor_optim.zero_grad()

        policy_loss = -self.critic((state_batch), self.actor((state_batch)))

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()

    def adv_preceive(self, state, action, mask, next_state, reward):
        experience = (state, action, mask, next_state, reward)
        self.episode_storage.append(experience)
        if len(self.episode_storage) >= self.episode_storage_end:
            self.replay_buffer.push(self.episode_storage[self.episode_storage_start: self.episode_storage_end])
            self.episode_storage_start += 1
            self.episode_storage_end += 1

    def reset_noise(self, i_episode=-1):
        # Todo: test later
        if i_episode != -1 and self.ou_noise:
            self.ou_noise.scale = (self.noise_scale - self.final_noise_scale) \
                                    * max(0, self.exploration_end - i_episode) / \
                                  self.exploration_end + self.final_noise_scale

        if self.param_noise:
            self.perturb_actor_parameters()

        self.ou_noise.reset()

    def perturb_actor_parameters(self):
        """Apply parameter noise to actor model, for exploration"""
        hard_update(self.actor_perturbed, self.actor)
        params = self.actor_perturbed.state_dict()
        for name in params:
            if 'ln' in name:
                pass
            param = params[name]
            param += torch.randn(param.shape).to(self.device) * self.param_noise.current_stddev

    def save_model(self, env_name, suffix='0', actor_path=None, critic_path=None, id='0', g='0'):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = 'models/dc_actor_{}_{}_{}_{}'.format(env_name, suffix, id, g)
        if critic_path is None:
            critic_path = 'models/dc_critic_{}_{}_{}_{}'.format(env_name, suffix, id, g)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path:
            self.actor.load_state_dict(torch.load(actor_path))
        if critic_path:
            self.critic.load_state_dict(torch.load(critic_path))

    def load_model_with_hard_update(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path:
            self.actor.load_state_dict(torch.load(actor_path))
            hard_update(self.actor_target, self.actor)
            hard_update(self.actor_perturbed, self.actor)
        if critic_path:
            self.critic.load_state_dict(torch.load(critic_path))
            hard_update(self.critic_target, self.critic)

    def load_full_model(self, paths):
        print('Loading full models')
        self.actor.load_state_dict(torch.load(paths[0]))
        self.actor_target.load_state_dict(torch.load(paths[1]))
        self.actor_perturbed.load_state_dict(torch.load(paths[2]))
        self.critic.load_state_dict(torch.load(paths[3]))
        self.critic_target.load_state_dict(torch.load(paths[4]))


    def update_param_noise(self):
        if self.param_noise and len(self.replay_buffer) >= self.batch_size:
            episode_transitions = self.replay_buffer.buffer[self.replay_buffer.position
                                                            -self.batch_size : self.replay_buffer.position]
            states = torch.cat([transition[0] for transition in episode_transitions], 0)
            unperturbed_actions = self.select_action(states)
            perturbed_actions = torch.cat([transition[1] for transition in episode_transitions], 0)

            if self.device == torch.device('cpu'):
                dist = distance_metric(perturbed_actions.numpy(), unperturbed_actions.numpy())
            else:
                dist = distance_metric(perturbed_actions.cpu().numpy(), unperturbed_actions.cpu().numpy())
            self.param_noise.adapt(dist)

    
    def follow_policy(self, state, action):
        predict_action = self.eval_action(state)

        if self.is_continue:
            return torch.equal(predict_action, action)
        else:
            assert self.env_name == 'MyEnv-10', 'specific the following code to adapt to your envs'
            for n_user in range(10):
                if predict_action[0, n_user * 2] * action[0, n_user * 2] < 0:
                    return False
                if int((predict_action[0, n_user * 2 + 1].float() + 1) * 2.5) != int((action[0, n_user * 2 + 1].float() + 1) * 2.5):
                    return False

        return True
    
    def reset_storage(self):
        while self.episode_storage_start < self.episode_storage_end - 1 and len(self.episode_storage) > 0:
            self.replay_buffer.push(self.episode_storage[self.episode_storage_start: self.episode_storage_end])
            self.episode_storage_start += 1

        self.episode_storage = []
        self.episode_storage_start = 0
        self.episode_storage_end = self.max_n

    def copy_model_mp(self, agent):
        hard_update(self.actor, agent.actor)
        hard_update(self.actor_target, agent.actor_target)
        hard_update(self.actor_perturbed, agent.actor_perturbed)

        hard_update(self.critic, agent.critic)
        hard_update(self.critic_target, agent.critic_target)

    def share_memory(self):
        self.actor.share_memory()
        self.actor_target.share_memory()
        self.actor_perturbed.share_memory()

        self.critic.share_memory()
        self.critic_target.share_memory()

    # for debug only
    def eval_performance(self):
        state = torch.tensor([self.eval_env.reset()]).float().to(self.device)
        episode_reward = 0
        for i_step in range(self.eval_env.spec.timestep_limit):
            action = self.eval_action(state)

            next_state, reward, done, _ = self.eval_env.step(action.cpu().numpy()[0])
            episode_reward += reward

            next_state = torch.tensor([next_state]).float().to(self.device)

            state = next_state
            if done:
                break
        return episode_reward

    def copy_actor_params(self, next_actor_params):
        for target_param, param in zip(self.actor.parameters(), next_actor_params):
            target_param.data.copy_(param)

    def copy_params(self, next_actor_params, next_actor_target_params, next_actor_perturbed_params,
                    next_critic_params,
                    next_critic_target_params):

        for target_param, param in zip(self.actor.parameters(), next_actor_params):
            target_param.data.copy_(param)

        for target_param, param in zip(self.actor_target.parameters(), next_actor_target_params):
            target_param.data.copy_(param)

        for target_param, param in zip(self.actor_perturbed.parameters(), next_actor_perturbed_params):
            target_param.data.copy_(param)

        for target_param, param in zip(self.critic.parameters(), next_critic_params):
            target_param.data.copy_(param)

        for target_param, param in zip(self.critic_target.parameters(), next_critic_target_params):
            target_param.data.copy_(param)

    def save_actor_params(self, env_name, suffix='', paths=None, id='0', g='0'):
        root_path = 'models/' + id + '_' + g + '/'
        if not os.path.exists(root_path):
            os.makedirs(root_path)

        if paths is None:
            actor_path = root_path + 'dc_actor_{}_{}_{}_{}'.format(env_name, suffix, id, g)
            actor_target_path = root_path + 'dc_actor_target_{}_{}_{}_{}'.format(env_name, suffix, id, g)
            actor_perturbed_path = root_path + 'dc_actor_perturbed_{}_{}_{}_{}'.format(env_name, suffix, id, g)
        else:
            actor_path = paths[0]
            actor_target_path = paths[1]
            actor_perturbed_path = paths[2]

        print('Saving actor models to {}'.format(actor_path))
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.actor_target.state_dict(), actor_target_path)
        torch.save(self.actor_perturbed.state_dict(), actor_perturbed_path)
        return actor_path, actor_target_path, actor_perturbed_path


    def save_critic_params(self, env_name, suffix='', paths=None, id='0', g='0'):
        root_path = 'models/' + id + '_' + g + '/'
        if not os.path.exists(root_path):
            os.makedirs(root_path)

        if paths is None:
            critic_path = root_path + 'dc_critic_{}_{}_{}_{}'.format(env_name, suffix, id, g)
            critic_target_path = root_path + 'dc_critic_target_{}_{}_{}_{}'.format(env_name, suffix, id, g)
        else:
            critic_path = paths[0]
            critic_target_path = paths[1]

        print('Saving critic models to {}'.format(critic_path))
        torch.save(self.critic.state_dict(), critic_path)
        torch.save(self.critic_target.state_dict(), critic_target_path)
        return critic_path, critic_target_path


def run_simple():

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str2bool, default=True)
    parser.add_argument('--env_name', type=str)
    parser.add_argument('--save', type=str2bool, default=False)
    parser.add_argument('--gamma', type=float, default=0.9, metavar='G')
    parser.add_argument('--tau', type=float, default=0.001, metavar='G')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N')
    parser.add_argument('--replay_size', type=int, default=100000, metavar='N')
    parser.add_argument('--replay_start_size', type=int, default=10000)
    parser.add_argument('--n_step', type=str2bool, default=False)
    parser.add_argument('--ou_noise', type=str2bool, default=True)
    parser.add_argument('--param_noise', type=str2bool, default=False)
    parser.add_argument('--load_path', type=str, default='0')
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--interval_exp', type=int, default=1000)

    args = parser.parse_args()

    assert args.env_name, 'Please specific the env name'


    ENV_NAME = args.env_name
    EPISODES = 100000
    TEST = 50

    total_numsteps = 0

    env = utils.makeFilteredEnv(gym.make(ENV_NAME))

    if args.gpu:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    writer = SummaryWriter()

    agent = Agent(env_name=ENV_NAME, obs_space=env.observation_space, act_space=env.action_space,
                  device=device, writer=writer, mode=None, gamma=args.gamma, batch_size=args.batch_size,
                  tau=args.tau, replay_size=args.replay_size, replay_start_size=args.replay_start_size,
                  n_step=args.n_step, ou_noise=args.ou_noise, param_noise=args.param_noise)

    agent.cuda()

    # agent.load_model('models/dc_actor_' + args.env_name + '_' + args.load_path,
    #                  'models/dc_critic_' + args.env_name + '_' + args.load_path)

    # agent.save_actor_params(args.env_name, g='0')
    # agent.save_critic_params(args.env_name, g='0')
    # agent.save_actor_params(args.env_name, g='1')
    # agent.save_critic_params(args.env_name, g='1')

    # env.monitor.start('experiments/' + ENV_NAME, force=True)

    rewards = []
    for i_episode in range(EPISODES):
        if args.n_step:
            agent.reset_storage()

        state = torch.tensor([env.reset()]).float().to(device)
        episode_reward = 0

        for i_step in range(env.spec.timestep_limit):
            action = agent.select_action(state).to(device)
            next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
            total_numsteps += 1
            episode_reward += reward

            mask = torch.tensor([not done]).float().to(device)
            next_state = torch.tensor([next_state]).float().to(device)
            reward = torch.tensor([reward]).float().to(device)

            # print(state, action, mask, next_state, reward)
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

        # if len(agent.replay_buffer) > args.interval_exp:
        #     print('save exp')
        #     exp_np = np.array(agent.replay_buffer.buffer)
        #     np.save('exps/exp_0_0_1000', exp_np)
        #     exit(1)

        writer.add_scalar('reward/train', episode_reward, i_episode)
        agent.update_param_noise()

        rewards.append(episode_reward)
        if i_episode % TEST == 0:
            state = torch.tensor([env.reset()]).float().to(device)
            episode_reward = 0
            for i_step in range(env.spec.timestep_limit):
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

            exit(1)
            # agent.save_model(args.env_name, suffix='0', g='1')

    env.monitor.close()


if __name__ == '__main__':
    run_simple()
