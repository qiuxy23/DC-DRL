import argparse

import torch
import torch.nn as nn
import numpy as np
import gym
import fastrand
import random
import math
import time
import os
import shutil
from conf.default import Settings

# from baselines import bench
# from baselines.common.atari_wrappers import make_atari, wrap_deepmind
# from baselines.common.vec_env import VecEnvWrapper
# from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
# from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
# from baselines.common.vec_env.vec_normalize import VecNormalize as VecNormalize_
# import dm_control2gym
# import roboschool
# import pybullet_envs
#
#
# # adapted from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
# def make_env(env_id, seed, rank, log_dir, allow_early_resets):
#     def _thunk():
#         if env_id.startswith('dm'):
#             _, domain, task = env_id.split('.')
#             env = dm_control2gym.make(domain_name=domain, task_name=task)
#         else:
#             env = gym.make(env_id)
#
#         is_atari = hasattr(gym.envs, 'atari') and isinstance(
#             env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
#         if is_atari:
#             env = make_atari(env_id)
#
#         env.seed(seed + rank)
#
#         if str(env.__class__.__name__).find('TimeLimit') >= 0:
#             env = TimeLimitMask(env)
#
#         if log_dir is not None:
#             env = bench.Monitor(
#                 env,
#                 os.path.join(log_dir, str(rank)),
#                 allow_early_resets=allow_early_resets)
#
#         if is_atari:
#             if len(env.observation_space.shape) == 3:
#                 env = wrap_deepmind(env)
#         elif len(env.observation_space.shape) == 3:
#             raise NotImplementedError(
#                 'CNN models work only for atari,\n'
#                 'please use a custom wrapper for a custom pixel input env.\n'
#                 'See wrap_deepmind for an example.')
#
#         # If the input has shape (W,H,3), wrap for PyTorch convolutions
#         # obs_shape = env.observation_space.shape
#         # not image
#         # if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
#         #     env = TransposeImage(env, op=[2, 0, 1])
#
#         return env
#
#     return _thunk
#
# def make_vec_envs(env_name,
#                   seed,
#                   num_processes,
#                   gamma,
#                   log_dir,
#                   device,
#                   allow_early_resets,
#                   num_frame_stack=None):
#     envs = [
#         make_env(env_name, seed, i, log_dir, allow_early_resets)
#         for i in range(num_processes)
#     ]
#
#     if len(envs) > 1:
#         envs = ShmemVecEnv(envs, context='fork')
#     else:
#         envs = DummyVecEnv(envs)
#
#     if len(envs.observation_space.shape) == 1:
#         if gamma is None:
#             envs = VecNormalize(envs, ret=False)
#         else:
#             envs = VecNormalize(envs, gamma=gamma)
#
#     envs = VecPyTorch(envs, device)
#
#     # if num_frame_stack is not None:
#     #     envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
#     # elif len(envs.observation_space.shape) == 3:
#     #     envs = VecPyTorchFrameStack(envs, 4, device)
#
#     return envs
#
# class VecNormalize(VecNormalize_):
#     def __init__(self, *args, **kwargs):
#         super(VecNormalize, self).__init__(*args, **kwargs)
#         self.training = True
#
#     def _obfilt(self, obs, update=True):
#         if self.ob_rms:
#             if self.training and update:
#                 self.ob_rms.update(obs)
#             obs = np.clip((obs - self.ob_rms.mean) /
#                           np.sqrt(self.ob_rms.var + self.epsilon),
#                           -self.clipob, self.clipob)
#             return obs
#         else:
#             return obs
#
#     def train(self):
#         self.training = True
#
#     def eval(self):
#         self.training = False
#
# class VecPyTorch(VecEnvWrapper):
#     def __init__(self, venv, device):
#         """Return only every `skip`-th frame"""
#         super(VecPyTorch, self).__init__(venv)
#         self.device = device
#         # TODO: Fix data types
#
#     def reset(self):
#         obs = self.venv.reset()
#         obs = torch.from_numpy(obs).float().to(self.device)
#         return obs
#
#     def step_async(self, actions):
#         if isinstance(actions, torch.LongTensor):
#             # Squeeze the dimension for discrete actions
#             actions = actions.squeeze(1)
#         actions = actions.cpu().numpy()
#         self.venv.step_async(actions)
#
#     def step_wait(self):
#         obs, reward, done, info = self.venv.step_wait()
#         obs = torch.from_numpy(obs).float().to(self.device)
#         reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
#         return obs, reward, done, info
#
# # Checks whether done was caused my timit limits or not
# class TimeLimitMask(gym.Wrapper):
#     def step(self, action):
#         obs, rew, done, info = self.env.step(action)
#         if done and self.env._max_episode_steps == self.env._elapsed_steps:
#             info['bad_transition'] = True
#
#         return obs, rew, done, info
#
#     def reset(self, **kwargs):
#         return self.env.reset(**kwargs)


def log_args(*lines):
    """
    func: to save output for debug
    :param lines:
    :return:
    """
    fo = open('out_' + str(Settings.postfix) + '.log', 'a')
    for line in lines:
        fo.write(str(line) + ' ')
    fo.write('\n')
    fo.close()


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias

class Evolution:
    """
        support a easier evaluation
    """
    def __init__(self, args):
        self.current_gen = 0
        self.population_size = args.n_population
        self.tournament_size = args.tournament_size
        self.num_elitists = int(args.elite_fraction * args.n_population)
        self.crossover_prob = args.crossover_prob
        self.mutation_prob = args.mutation_prob

        if self.num_elitists < 1:
            self.num_elitists = 1

        self.rl_policy_index = None
        self.selection_stats = {'elite': 0, 'selected': 0, 'discarded': 0, 'total': 0.0000001}

        self.super_mut_prob = 0.05
        self.reset_prob = self.super_mut_prob + 0.05
        self.super_mut_strength = 1
        self.mut_strength = 0.1

    def torch_next_g(self, agent_population, all_fitness):
        index_rank = self.list_argsort(all_fitness)
        index_rank.reverse()
        elitist_index = index_rank[:self.num_elitists]

        elitists = []
        denominator = 0
        for i in elitist_index:
            elitists.append(agent_population[i])
            denominator += all_fitness[i]

        next_actor_params, next_actor_target_params, next_actor_perturbed_params, next_critic_params, \
            next_critic_target_params = self.generation_distribution(elitist_index, agent_population,  all_fitness, denominator)

        losers = []
        for i in range(self.population_size):
            if i not in elitist_index:
                losers.append(agent_population[i])

        for loser in losers:
            if random.random() < self.crossover_prob:
                loser.copy_params(next_actor_params, next_actor_target_params, next_actor_perturbed_params, next_critic_params,
                              next_critic_target_params)

        # for i in range(self.population_size):
        #     if random.random() < self.mutation_prob:
        #         mut_param = self.mutate_gau(agent_population[i])
        #         agent_population[i].copy_actor_params(mut_param)
        for loser in losers:
            if random.random() < self.mutation_prob:
                mut_param = self.mutate_gau(loser)
                loser.copy_actor_params(mut_param)



    def generation_distribution(self, elitist_index, agent_population, all_fitness, denominator):

        next_actor_params = []
        for counter, i_elit in enumerate(elitist_index):
            for idx, param in enumerate(agent_population[i_elit].actor.parameters()):
                if counter == 0:
                    next_actor_params.append(param.data * all_fitness[i_elit] / denominator)
                else:
                    next_actor_params[idx] = next_actor_params[idx] + param.data * all_fitness[i_elit] / denominator

        next_actor_target_params = []
        for counter, i_elit in enumerate(elitist_index):
            for idx, param in enumerate(agent_population[i_elit].actor_target.parameters()):
                if counter == 0:
                    next_actor_target_params.append(param.data * all_fitness[i_elit] / denominator)
                else:
                    next_actor_target_params[idx] = next_actor_target_params[idx] + param.data * all_fitness[
                        i_elit] / denominator

        next_actor_perturbed_params = []
        for counter, i_elit in enumerate(elitist_index):
            for idx, param in enumerate(agent_population[i_elit].actor_perturbed.parameters()):
                if counter == 0:
                    next_actor_perturbed_params.append(param.data * all_fitness[i_elit] / denominator)
                else:
                    next_actor_perturbed_params[idx] = next_actor_perturbed_params[idx] + param.data * all_fitness[
                        i_elit] / denominator

        next_critic_params = []
        for counter, i_elit in enumerate(elitist_index):
            for idx, param in enumerate(agent_population[i_elit].critic.parameters()):
                if counter == 0:
                    next_critic_params.append(param.data * all_fitness[i_elit] / denominator)
                else:
                    next_critic_params[idx] = next_critic_params[idx] + param.data * all_fitness[i_elit] / denominator

        next_critic_target_params = []
        for counter, i_elit in enumerate(elitist_index):
            for idx, param in enumerate(agent_population[i_elit].critic_target.parameters()):
                if counter == 0:
                    next_critic_target_params.append(param.data * all_fitness[i_elit] / denominator)
                else:
                    next_critic_target_params[idx] = next_critic_target_params[idx] + param.data * all_fitness[
                        i_elit] / denominator
        return next_actor_params, next_actor_target_params, next_actor_perturbed_params, next_critic_params, \
            next_critic_target_params



    # a more sophisticated design for the next generation operation
    def next_g(self, actors, all_fitness):

        index_rank = self.list_argsort(all_fitness)
        index_rank.reverse()
        elitist_index = index_rank[:self.num_elitists]

        offsprings = self.selection_tournament(index_rank, num_offsprings=len(index_rank) - self.num_elitists,
                                               tournament_size=self.tournament_size)

        losers = []
        for i in range(self.population_size):
            if i in offsprings or i in elitist_index:
                continue
            else:
                losers.append(i)
        random.shuffle(losers)

        if self.rl_policy_index != None:
            self.selection_stats['total'] += 1.0
            if self.rl_policy_index in elitist_index:
                self.selection_stats['elite'] += 1.0
            elif self.rl_policy_index in offsprings:
                self.selection_stats['selected'] += 1.0
            elif self.rl_policy_index in losers:
                self.selection_stats['discarded'] += 1.0
            self.rl_policy_index = None


        new_elitists = []
        for i_elitist in elitist_index:
            try:
                replace = losers.pop(0)
            except:
                replace = offsprings.pop(0)
            new_elitists.append(replace)
            self.hard_clone(master_actor=actors[i_elitist], replace_actor=actors[replace])

        if len(losers) % 2 != 0:
            losers.append(losers[fastrand.pcg32bounded(len(losers))])
        for i, j in zip(losers[0::2], losers[1::2]):
            off_i = random.choice(new_elitists)
            off_j = random.choice(offsprings)
            self.hard_clone(master_actor=actors[off_i], replace_actor=actors[i])
            self.hard_clone(master_actor=actors[off_j], replace_actor=actors[j])
            self.crossover(actors[i], actors[j])

        for i, j in zip(offsprings[0::2], offsprings[1::2]):
            if random.random() < self.crossover_prob:
                self.crossover(actors[i], actors[j])

        for i in range(self.population_size):
            if i not in new_elitists:
                if random.random() < self.mutation_prob:
                    self.mutate(actors[i])

        return new_elitists[0]


    def list_argsort(self, seq):
        return sorted(range(len(seq)), key=seq.__getitem__)

    def selection_tournament(self, index_rank, num_offsprings, tournament_size):
        total_choices = len(index_rank)
        offsprings = []
        for i in range(num_offsprings):
            winner = np.min(np.random.randint(total_choices, size=tournament_size))
            offsprings.append(index_rank[winner])

        offsprings = list(set(offsprings))  # get unique offsprings
        if len(offsprings) % 2 != 0:
            offsprings.append(offsprings[fastrand.pcg32bounded(len(offsprings))])
        return offsprings  # index

    def hard_clone(self, master_actor, replace_actor):
        replace_actor.copy_model(master_actor)
        # parameter_list = master_actor.get_parameters()
        # replace_actor.set_parameters(parameter_list)

    def crossover(self, actor1, actor2):

        ori_parameters1, ori_parameters2 = actor1.get_parameters(), actor2.get_parameters()

        for param1, param2 in zip(ori_parameters1, ori_parameters2):

            if len(param1.shape) == 2: # weight
                num_variables = param1.shape[0]
                num_cross_overs = fastrand.pcg32bounded(num_variables * 2)
                for i in range(num_cross_overs):
                    if random.random() < 0.5:
                        ind_cr = fastrand.pcg32bounded(param1.shape[0])
                        param1[ind_cr, :] = param2[ind_cr, :]
                    else:
                        ind_cr = fastrand.pcg32bounded(param1.shape[0])
                        param2[ind_cr, :] = param1[ind_cr, :]

            elif len(param1.shape) == 1: # bias
                num_variables = param1.shape[0]
                num_cross_overs = fastrand.pcg32bounded(num_variables)
                for i in range(num_cross_overs):
                    if random.random() < 0.5:
                        ind_cr = fastrand.pcg32bounded(param1.shape[0])
                        param1[ind_cr] = param2[ind_cr]
                    else:
                        ind_cr = fastrand.pcg32bounded(param1.shape[0])
                        param2[ind_cr] = param1[ind_cr]

        actor1.set_parameters(ori_parameters1)
        actor2.set_parameters(ori_parameters2)

    def mutate(self, actor):

        num_mutation_frac = 0.1
        super_mut_prob = 0.05
        reset_prob = super_mut_prob + 0.05
        super_mut_strength = 10
        mut_strength = 0.1

        ori_weights = actor.get_parameters_weights()

        for weight in ori_weights:
            if len(weight.shape) == 2:
                num_weights = weight.shape[0] * weight.shape[1]
                num_mutations = fastrand.pcg32bounded(int(math.ceil(num_mutation_frac * num_weights)))

                for _ in range(num_weights):
                    ind_dim1 = fastrand.pcg32bounded(weight.shape[0])
                    ind_dim2 = fastrand.pcg32bounded(weight.shape[1])
                    random_num = random.random()

                    if random_num < super_mut_prob:  # Super Mutation probability
                        weight[ind_dim1, ind_dim2] += random.gauss(0, super_mut_strength * weight[ind_dim1, ind_dim2])
                    elif random_num < reset_prob:
                        weight[ind_dim1, ind_dim2] = random.gauss(0, 1)
                    else:  # mutauion even normal
                        weight[ind_dim1, ind_dim2] += random.gauss(0, mut_strength * weight[ind_dim1, ind_dim2])

                    weight[ind_dim1, ind_dim2] = self.regularize_weight(weight[ind_dim1, ind_dim2], 100000)

        actor.set_parameters_weights(ori_weights)


    def mutate_gau(self, agent): # model can be online actor or critic

        mut_params = []
        for param in agent.actor.parameters():
            temp = torch.normal(mean=0, std=param)
            out = torch.add(param, temp)
            mut_params.append(out)
        return mut_params

            # p_size = param.size()
            # if len(p_size) == 2:
            #     num_weights = p_size[0] * p_size[1]
            #
            #     for _ in range(int(num_weights/10)):
            #         ind_dim1 = fastrand.pcg32bounded(p_size[0])
            #         ind_dim2 = fastrand.pcg32bounded(p_size[1])
            #         random_num = random.random()
            #         if random_num < self.super_mut_prob:  # Super Mutation probability
            #             param[ind_dim1, ind_dim2] += random.gauss(0, self.super_mut_strength * param[ind_dim1, ind_dim2])
            #         elif random_num < self.reset_prob:
            #             param[ind_dim1, ind_dim2] = random.gauss(0, 1)
            #         else:  # mutauion even normal
            #             param[ind_dim1, ind_dim2] += random.gauss(0,self. mut_strength * param[ind_dim1, ind_dim2])
            #
            # elif len(p_size) == 1:
            #     num_bias = p_size[0]
            #     for _ in range(int(num_bias/10)):
            #         ind_dim = fastrand.pcg32bounded(num_bias)
            #
            #         random_num = random.random()
            #         if random_num < self.super_mut_prob:  # Super Mutation probability
            #             param[ind_dim] += random.gauss(0, self.super_mut_strength * param[ind_dim])
            #         elif random_num < self.reset_prob:
            #             param[ind_dim] = random.gauss(0, 1)
            #         else:  # mutauion even normal
            #             param[ind_dim] += random.gauss(0, self.mut_strength * param[ind_dim])



    def regularize_weight(self, weight, mag):
        if weight > mag:
            weight = mag
        elif weight < -mag:
            weight = -mag
        return weight


def makeFilteredEnv(env):
    """
        a function used to normalized the env output
    """
    acsp = env.action_space
    obsp = env.observation_space

    # print('obsp:', obsp)

    if not type(acsp) == gym.spaces.box.Box:
        raise RuntimeError('Environment with continous action space (i.e. Box) required')
    if not type(obsp) == gym.spaces.box.Box:
        raise RuntimeError('Environment with continous observation space (i.e. Box) required')

    env_type = type(env)

    class FilteredEnv(env_type):
        def __init__(self):
            self.__dict__.update(env.__dict__)

            if np.any(obsp.high < 1e10):
                h = obsp.high
                l = obsp.low
                sc = h-l
                self.o_c = (h+l)/2.
                self.o_sc = sc/2.
            else:
                self.o_c = np.zeros_like(obsp.high)
                self.o_sc = np.ones_like(obsp.high)

            h = acsp.high
            l = acsp.low
            sc = h-l
            self.a_c = (h+l)/2.
            self.a_sc = sc / 2.

            self.r_sc = 0.1
            self.r_c = 0.

            self.observation_space = gym.spaces.Box(self.filter_observation(obsp.low),
                                              self.filter_observation(obsp.high))
            self.action_space = gym.spaces.Box(-np.ones_like(acsp.high), np.ones_like(acsp.high))

            def assertEqual(a, b):
                assert np.all(a == b), "{} != {}".format(a, b)

            assertEqual(self.filter_action(self.action_space.low), acsp.low)
            assertEqual(self.filter_action(self.action_space.high), acsp.high)

        def filter_observation(self, obs):
            return (obs - self.o_c) / self.o_sc

        def filter_action(self, action):
            return self.a_sc * action + self.a_c

        def filter_reward(self, reward):
            return self.r_sc * reward + self.r_c

        def step(self, action):
            ac_f = np.clip(self.filter_action(action), self.action_space.low, self.action_space.high)
            obs, reward, term, info = env_type.step(self, ac_f) # super function
            obs_f = self.filter_observation(obs)

            return obs_f, reward, term, info

    fenv = FilteredEnv()

    print('True action space: ' + str(acsp.low) + ', ' + str(acsp.high))
    print('True state space: ' + str(obsp.low) + ', ' + str(obsp.high))
    print('Filtered action space: ' + str(fenv.action_space.low) + ', ' + str(fenv.action_space.high))
    print('Filtered state space: ' + str(fenv.observation_space.low) + ', ' + str(fenv.observation_space.high))

    return fenv


class NormalizedActions(gym.ActionWrapper):
    """
        a common class used to normalized action to [0, 2]
        but we recomment to do it directly in the env output
    """
    def _action(self, action):
        action = (action + 1) / 2  # [-1, 1] => [0, 1]
        action *= (self.action_space.high - self.action_space.low)
        action += self.action_space.low
        return action

    def _reverse_action(self, action):
        action -= self.action_space.low
        action /= (self.action_space.high - self.action_space.low)
        action = action * 2 - 1
        return action

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_timely_exp_dirs(root_path, cycles_str):

    all_dirs = os.listdir(root_path)

    timely_paths = []
    timely_rm = []
    for i_dir in all_dirs:
        flag = False
        for cycle_str in cycles_str:
            if i_dir.endswith(cycle_str):
                sub_temp = os.listdir(root_path + i_dir)
                if len(sub_temp) > 0:
                    timely_paths.append(root_path + i_dir + '/' + sub_temp[0])
                    timely_rm.append(root_path + i_dir)
                    flag = True
                    break
        if not flag:
            try:
                shutil.rmtree(root_path + i_dir)
            except:
                print('PermissionError')

    return timely_paths, timely_rm

