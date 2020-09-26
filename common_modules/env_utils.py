import torch
import numpy as np
import gym
from gym.spaces import Box
import copy
import math
import random

def init_pos(seed, n_users, cover_range):
    np.random.seed(seed)
    return np.random.randint(low=-cover_range, high=cover_range, size=n_users * 2)


def lower_ver(cur, target):
    cur = cur.split('.')
    cur = [int(v) for v in cur]

    target = target.split('.')
    target = [int(v) for v in target]

    for v1,v2 in zip(cur, target):
        if v1 > v2:
            return False
        elif v1 < v2:
            return True
    return True


def get_space(obs_len, acs_len):
    obsp_low = np.zeros(obs_len)
    obsp_high = np.ones(obs_len)

    if lower_ver(gym.__version__, '0.9.0'):
        observation_space = Box(low=obsp_low, high=obsp_high)
    else:
        observation_space = Box(low=obsp_low, high=obsp_high, dtype=float)

    acsp_low = -np.ones(acs_len)
    acsp_high = np.ones(acs_len)

    if lower_ver(gym.__version__, '0.9.0'):
        action_space = Box(low=acsp_low, high=acsp_high)
    else:
        action_space = Box(low=acsp_low, high=acsp_high, dtype=float)

    return observation_space, action_space

def all_zeros(n_users):
    obs_users = []
    for i_user in range(n_users):
        obs_users.append([0 for _ in range(7)])
    obs_users.append([0, 0, 0])
    return obs_users

def get_normalized_state(obs_users, n_users):
    state = copy.deepcopy(obs_users)
    for i_user in range(n_users):
        state[i_user][1] /= 2.
        state[i_user][2] = (state[i_user][2] + 80.) / 160.
        state[i_user][3] = (state[i_user][3] + 80.) / 160.
        state[i_user][5] /= 5.
        state[i_user][6] /= 1e4

    state[n_users][0] /= 8.
    state[n_users][2] /= 50.
    return state

def get_normalized_state_tb(obs_users, n_users):
    # note the modification here will only change the exp store in replay buffer
    # which has no effect on the agent-envirnoment interaction and the reward
    state = copy.deepcopy(obs_users)
    for i_user in range(n_users):
        state[i_user][0] /= 2.
        state[i_user][1] /= 500.
        state[i_user][2] = (state[i_user][2] + 80.) / 160.
        state[i_user][3] = (state[i_user][3] + 80.) / 160.
        state[i_user][5] /= 500.
        state[i_user][6] /= 1e4

    state[n_users][0] /= 20.
    state[n_users][1] /= 500.
    state[n_users][2] /= 50.
    return state

def get_counter(action, n_users):
    counter = 0
    for i_user in range(n_users):
        if action[i_user * 2] >= 0.:
            counter += 1
    return counter


def get_transmit_power(uid, pos, args):
    distance = math.sqrt(pos[2 * uid] * pos[2 * uid] + pos[2 * uid + 1] * pos[2 * uid + 1])
    return args.power_tr * math.pow(distance, -args.alpha_p)


def get_rate(uid, pos, power_tr, args, action, n_users):
    power_i = 0
    for n_user in range(n_users):
        if int((action[n_user * 2 + 1] + 1) * 2.5) != int((action[uid * 2 + 1] + 1) * 2.5):
            power_i += get_transmit_power(n_user, pos, args)
            power_i = 0
    snr = power_tr / (power_i + args.noise)
    return args.c_band * math.log2(1 + snr) /8/1024


def step(obs_users, action, n_users, pos, args, counter):
    reward = 0
    done = False

    for i_user in range(n_users):
        cost = 0
        if action[i_user * 2] < 0:
            time_l = obs_users[i_user][4] / obs_users[i_user][0]
            cost += obs_users[i_user][4] * args.unit_local

            if time_l + obs_users[i_user][1] > obs_users[i_user][5]:
                cost += args.Theta

            obs_users[i_user][1] = max(0, obs_users[i_user][1] + time_l - args.len_epoch)

        elif action[i_user * 2] >= 0:
            power_o = get_transmit_power(i_user, pos, args)
            rate_o = get_rate(i_user, pos, power_o, args, action, n_users)
            time_o = obs_users[i_user][6] / rate_o
            time_e = obs_users[i_user][4] / obs_users[n_users][0] * counter

            cost += args.power_tr * time_o * args.unit_j
            cost += obs_users[i_user][4] * args.unit_edge

            if time_o + time_e + obs_users[n_users][1] > obs_users[i_user][5]:
                cost += args.Theta

            obs_users[i_user][1] = max(0, obs_users[i_user][1] - args.len_epoch)
            obs_users[n_users][1] += (time_e / counter)

        reward -= cost

    obs_users[n_users][1] = max(0, obs_users[n_users][1] - args.len_epoch)
    for i_user in range(n_users):
        obs_users[i_user][4] = random.uniform(0.6, 0.8)
        obs_users[i_user][5] = 2
        obs_users[i_user][6] = random.uniform(5000, 8000)
    args.count += 1
    obs_users[n_users][2] = args.count

    # if args.count >= 50:
    #     done = True

    return reward, done


def get_denominator_fortb():
    '''
    as the testbed cpu cycles are much large than the setting in simulations,
    we divide the total cycles with 1 Gcycle
    this can be consider that perfroming face recognition task for a video consist of several epoches
    note that all algorithm use the same denominator
    :return: the total number of epochs
    '''
    cycles = np.load('../conf/cycles.npy')
    denominator = sum(cycles)
    # print(denominator/23.) # to determine the overall penalty for each task, as we consider it as a whole
    return denominator


def step_tb(obs_users, action, n_users, pos, args, counter, next_cycle, next_size):
    reward = 0
    done = False

    for i_user in range(n_users):
        cost = 0
        if action[i_user * 2] < 0:
            time_l = obs_users[i_user][4] * 400 / obs_users[i_user][0]

            cost += obs_users[i_user][4] * 400 * args.unit_local

            if time_l + obs_users[i_user][1] > obs_users[i_user][5]:
                cost += args.Theta
                args.fail_times += 1

            obs_users[i_user][1] = max(0, obs_users[i_user][1] + time_l - args.len_epoch)

        elif action[i_user * 2] >= 0:
            power_o = get_transmit_power(i_user, pos, args)
            rate_o = get_rate(i_user, pos, power_o, args, action, n_users)
            time_o = obs_users[i_user][6] / rate_o
            time_e = obs_users[i_user][4] * 400 / obs_users[n_users][0] * counter

            cost += args.power_tr * time_o * args.unit_j
            cost += obs_users[i_user][4] * 400 * args.unit_edge

            if time_o + time_e + obs_users[n_users][1] > obs_users[i_user][5]:
                cost += args.Theta
                args.fail_times += 1

            obs_users[i_user][1] = max(0, obs_users[i_user][1] - args.len_epoch)
            obs_users[n_users][1] += (time_e / counter)

        reward -= cost

    obs_users[n_users][1] = max(0, obs_users[n_users][1] - args.len_epoch)
    for i_user in range(n_users):
        obs_users[i_user][4] = next_cycle/400
        obs_users[i_user][5] = args.ddl
        obs_users[i_user][6] = next_size/1024
    args.count += 1
    obs_users[n_users][2] = args.count

    # if args.count >= 50:
    #     done = True

    # note that we divide 1000. is only used to limit the rewards to a small range, as in [-1, 1]
    # this helps dnn to learning
    # while should remember to mul 1000. when process the result
    return reward/200., done

def get_dis(uid1, uid2, pos):
    x_pow = (pos[2 * uid1] - pos[2 * uid2]) * (pos[2 * uid1] - pos[2 * uid2])
    y_pow = (pos[2 * uid1 + 1] - pos[2 * uid2 + 1]) * (pos[2 * uid1 + 1] - pos[2 * uid2 + 1])
    return math.sqrt(x_pow + y_pow)


if __name__ == '__main__':
    get_denominator_fortb()