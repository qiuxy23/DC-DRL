import torch
import argparse
import math
from collections import namedtuple
from itertools import count
from tensorboardX import SummaryWriter
from datetime import datetime
import gym
import time
import numpy as np
from gym import wrappers
import copy
import torch.multiprocessing as torch_mp
import torch.multiprocessing.queue as mq
from common_modules.replay_buffer import ReplayBufferM, Transition
import os

from common_modules.agent import Agent
from common_modules.utils import *
from common_modules.utils import str2bool

def init_agent(args, device, writer, idx):

    # env = NormalizedActions(gym.make(args.env_name))
    # env = makeFilteredEnv(gym.make(ENV_NAME))
    env = gym.make(args.env_name)

    if args.save_path == '':
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        args.save_path = months[datetime.now().month] + str(datetime.now().day) + '_' + str(datetime.now().hour) +\
                         '-' + str(datetime.now().minute) + '-' + str(datetime.now().second) + str(idx)
    else:
        args.save_path += str(idx)

    assert np.all(env.action_space.low == -1), 'change the agent for action clamp'
    assert np.all(env.action_space.high == 1), 'change the agent for action clamp'

    agent = Agent(env_name=args.env_name, obs_space=env.observation_space,
                  act_space=env.action_space, device=device, writer=
                  writer, mode=None, eval_env=env, gamma=args.gamma, tau=
                  args.tau, replay_size=args.replay_size, batch_size=args.batch_size,
                  replay_start_size=args.replay_start_size, n_step=args.n_step,
                  ou_noise=args.ou_noise, param_noise=args.param_noise,
                  noise_scale=args.noise_scale, final_noise_scale=args.final_noise_scale,
                  updates_per_step=args.updates_per_step)

    if args.load_path != '':
        agent.load_model('models/ddpg_actor_' + args.env_name + '_' + args.load_path,
                         'models/ddpg_critic_' + args.env_name + '_' + args.load_path)

    agent.cuda()
    agent.share_memory()
    return agent

def train_agent(args, agent, idx_process, idx_population, generation, device, fitness_values, collect_queue, done_signals, share_signals):

    collect_queue.cancel_join_thread()

    # state = torch.tensor([agent.eval_env.reset()]).float().to(device)
    # episode_reward = 0
    # for i_step in range(agent.eval_env.spec.timestep_limit):
    #     action = agent.eval_action(state)
    #
    #     next_state, reward, done, _ = agent.eval_env.step(action.cpu().numpy()[0])
    #     episode_reward += reward
    #
    #     next_state = torch.tensor([next_state]).float().to(device)
    #
    #     state = next_state
    #     if done:
    #         break

    print('start to train agent {} at generatin {}, total episode {}, total timestep {}, init reward {}, init replay buffer {}'.format
          (idx_population, generation, generation * args.local_interval, generation * args.local_interval * agent.eval_env.spec.timestep_limit,
           None, len(agent.replay_buffer)))
    rewards = []

    # for i_episode in range(args.num_episodes+1):
    for i_episode in range(args.local_interval):
        state = torch.tensor([agent.eval_env.reset()]).float().to(device)

        episode_reward = 0
        for i_step in range(agent.eval_env.spec.timestep_limit):
            action = agent.select_action(state).to(device)
            next_state, reward, done, _ = agent.eval_env.step(action.cpu().numpy()[0])
            episode_reward += reward

            mask = torch.tensor([not done]).float().to(device)
            next_state = torch.tensor([next_state]).float().to(device)
            reward = torch.tensor([reward]).float().to(device)

            collect_queue.put((state, action, mask, next_state, reward))
            # agent.preceive(state, action, mask, next_state, reward)

            # can consider to be decouple from the experience collection
            if len(agent.replay_buffer) > agent.replay_start_size:
                for _ in range(agent.updates_per_step):
                    batch = agent.replay_buffer.sample(agent.batch_size)
                    # batch = Transition(*zip(*transitions))
                    value_loss, policy_loss = agent.update_parameters(batch)

                    if agent.writer:
                        agent.writer.add_scalar('loss/value', value_loss, agent.updates)
                        agent.writer.add_scalar('loss/policy', policy_loss, agent.updates)

                    agent.updates += 1

            state = next_state

            if done:
                agent.reset_noise(i_episode)
                break

        # if agent.writer:
        #     writer.add_scalar('reward/train', episode_reward, i_episode)

        agent.update_param_noise() # Update param_noise based on distance metric

        rewards.append(episode_reward)

    state = torch.tensor([agent.eval_env.reset()]).float().to(device)
    episode_reward = 0
    for i_step in range(agent.eval_env.spec.timestep_limit):
        action = agent.eval_action(state)

        next_state, reward, done, _ = agent.eval_env.step(action.cpu().numpy()[0])
        episode_reward += reward

        next_state = torch.tensor([next_state]).float().to(device)

        state = next_state
        if done:
            break

    # if agent.writer:
    #     writer.add_scalar('reward/test', episode_reward, i_episode)

    rewards.append(episode_reward)
    # print('Episode: {}, total numsteps: {}, reward: {}, average reward: {}, index: {}, time: {}'.format
    #       (generation * args.local_interval, generation * args.local_interval * agent.eval_env.spec.timestep_limit,
    #        rewards[-1], np.mean(rewards[-10:]), idx, time.time()))

    fitness_values[idx_population] = np.mean(rewards)
    done_signals[idx_population] = 1

    while share_signals.value == 0:
        time.sleep(1)

def mp_agent(args):

    assert True, 'annotate the code in agent preceive'

    if args.seed != -1:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    if args.gpu and False:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    # TensorboardX is incompatible with asynchronous event writing:
    # https://github.com/lanpa/tensorboardX/issues/123
    # writer = SummaryWriter()
    writer = None

    agent_population = []
    for i_population in range(args.n_population):
        agent_i = init_agent(args, device, writer, i_population)
        agent_population.append(agent_i)

    best_history_agent = init_agent(args, device, writer, args.n_population)
    best_history = -100000

    fitness_values = [0. for i in range(args.n_process)]
    fitness_values = torch_mp.Array('f', fitness_values)
    done_signals = torch_mp.Array('i', [0 for i in range(args.n_process)])  # as a compromise
    share_signals = torch_mp.Value('i', 0)
    collected_exp = []
    evo = Evolution(args)

    for i_process in range(args.n_process):  # change to population:
        q = torch_mp.Queue()  # change according to local_interval
        collected_exp.append(q)

    for i_generation in range(args.n_generation):
        processes = []
        done_signals[:] = [0 for i in range(args.n_process)]
        share_signals.value = 0

        for i_process in range(args.n_process):
            p = torch_mp.Process(target=train_agent, args=(args, agent_population[i_process], i_process, i_process,
                                                           i_generation, device, fitness_values,
                                                           collected_exp[i_process], done_signals, share_signals))
            p.start()
            processes.append(p)

        # for p in processes:
        #     p.join()

        while not all(done_signals[:]):
            # continues to check
            time.sleep(1)

        for i_process in range(args.n_process):
            q = collected_exp[i_process]
            q_counter = 0
            while q.qsize() > 0:
                q_counter += 1
                state, action, mask, next_state, reward = q.get()
                for agent_i in agent_population:
                    agent_i.preceive(state, action, mask, next_state, reward)
                    if q_counter % agent_i.eval_env.spec.timestep_limit == 0:
                        agent_i.reset_storage()

        share_signals.value = 1
        time.sleep(2)  # wait for the subprocess to close
        all_fitness = fitness_values[:]

        best_pop_fitness = max(all_fitness)
        best_index = all_fitness.index(max(all_fitness))

        if best_history < best_pop_fitness:
            best_history = best_pop_fitness
            best_history_agent.copy_model_mp(agent_population[best_index])

        print('eval_performance:', best_history_agent.eval_performance())

        for i_fitness in range(len(all_fitness)):  # manual syncs
        # change to your own environment
            if all_fitness[i_fitness] < best_history + best_history * 0.1:
                agent_population[i_fitness].copy_model_mp(best_history_agent)

        # evo.next_g(agent_population, all_fitness)
        evo.torch_next_g(agent_population, all_fitness)

    if writer:
        writer.close()

    # for i_population in range(args.n_population):
    #     agent_population[i_population].eval_env.close()
    #     agent_population[i_population].save_model(args.env_name, args.save_path) # Todo: specific

