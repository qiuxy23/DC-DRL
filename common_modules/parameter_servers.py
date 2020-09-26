# -*- coding: utf-8 -*-
import torch
import socket
import time
import argparse
import threading
import _thread
import os
import hashlib
import pickle
import struct
import shutil
import gym
import redis
import torch.nn.functional as F

import common_modules.utils as utils
from common_modules.env_utils import get_space
from common_modules.nets import Actor, Critic
from conf.default import Settings

def merge_model(models, models_holder, model_args, target_path, target_names):
    # actor_paths, at_paths, ap_paths, critic_paths, ct_paths
    print('merge length:', model_args['len'])
    if model_args['len'] == 0:
        return
    elif model_args['len'] == 1:
        print(model_args['actor_path'])
        # only one model, direct move
        actor_path, at_path, ap_path, critic_path, ct_path = \
            model_args['actor_path'][0], model_args['at_path'][0], model_args['ap_path'][0], \
            model_args['critic_path'][0], model_args['ct_path'][0]
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        shutil.copy(actor_path, target_path + target_names[0])
        shutil.copy(at_path, target_path + target_names[1])
        shutil.copy(ap_path, target_path + target_names[2])
        shutil.copy(critic_path, target_path + target_names[3])
        shutil.copy(ct_path, target_path + target_names[4])
    else:
        print(model_args['actor_path'])
        fitnesses = model_args['fitness']
        denominator = sum(fitnesses)
        keys = ['actor_path', 'at_path', 'ap_path', 'critic_path', 'ct_path']
        for i, model_holder in enumerate(models_holder):
            model_holder.load_state_dict(torch.load(model_args[keys[i]][0]))

        # for model, model_holder in zip(models, models_holder):

        res_params = [[], [], [], [], []]
        for i_model in range(model_args['len']):
            for i_key, key in enumerate(keys):
                models_holder[i_key].load_state_dict(torch.load(model_args[key][i_model]))

                if i_model == 0:
                    for idx_param, param in enumerate(models_holder[i_key].parameters()):
                        res_params[i_key].append(param.data * fitnesses[i_model] / denominator)
                else:
                    for idx_param, param in enumerate(models_holder[i_key].parameters()):
                        res_params[i_key][idx_param] = res_params[i_key][idx_param] + param.data * fitnesses[i_model] / denominator


        for idx in range(len(models)):
            for target_param, param in zip(models[idx].parameters(), res_params[idx]):
                target_param.data.copy_(param)

        if not os.path.exists(target_path):
            os.makedirs(target_path)

        for idx in range(len(models)):
            torch.save(models[idx].state_dict(), target_path + target_names[idx])



def tcplink(sock, addr, args):
    print('Accept new connection from', addr)
    sock.send(b'established')
    while True:
        data = sock.recv(2048)
        if data == b'exit' or not data:
            break
        if data.startswith(b'#info#'):
            str_info = data[6:]
            info = pickle.loads(str_info)
            print(info)
            sock.send(b'ack info')

            if info['type'] == 'model':
                size = info['size']

                save_path = args.root_path + str(info['c_id']) + '_' + str(info['g']) + '/'
                check_exist(save_path, info['name'])

                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                if size > 0:
                    r = redis.Redis(Settings.redis_host, Settings.redis_port)
                    n_g = int(r.hget('hash', 'n_g'))
                    while size > 0:
                        if size >= 4096:
                            d = sock.recv(4096)
                            write_file(save_path, info['name'], d)
                            size -= 4096
                        else:
                            d = sock.recv(size)
                            write_file(save_path, info['name'], d)
                            size = 0
                    sock.send(b'finish')

                    f_key = 'f_' + str(info['c_id']) + '_' + str(info['g'])
                    p_key = 'p_' + str(info['c_id']) + '_' + str(info['g']) + '_' + str(info['mt'])
                    if n_g < int(info['g']):
                        r.hset('hash', 'n_g', str(info['g']))
                    r.hset('hash', f_key, str(info['f']))
                    r.hset('hash', p_key, save_path + str(info['name']))
                    r.close()
                else:
                    sock.send(b'file size is 0')
            elif info['type'] == 'join':
                print('join')
                client_id = info['c_id']
                r = redis.Redis(Settings.redis_host, Settings.redis_port)
                r.sadd('agent_list', client_id)
                r.close()
            elif info['type'] == 'exit':
                client_id = info['c_id']
                r = redis.Redis(Settings.redis_host, Settings.redis_port)
                if r.sismember('agent_list', client_id):
                    r.srem('agent_list', client_id)
                r.close()
            elif info['type'] == 'request':
                cur_fit = info['f']
                r = redis.Redis(Settings.redis_host, Settings.redis_port)
                g = int(r.hget('hash', 'g'))
                f_avg = float(r.hget('hash', 'f_avg'))
                if f_avg > cur_fit:
                    sock.send(b'yes')
                    p_keys = ['p_avg_' + str(g) + '_a',
                              'p_avg_' + str(g) + '_at',
                              'p_avg_' + str(g) + '_ap',
                              'p_avg_' + str(g) + '_c',
                              'p_avg_' + str(g) + '_ct']

                    paths = []
                    for p_key in p_keys:
                        paths.append(r.hget('hash', p_key).decode())

                    for path in paths:
                        f = open(path, 'rb')

                        size = os.path.getsize(path)
                        print('Sent file size:', size)
                        data = b''
                        eachCount = 1024
                        tmp = f.read(eachCount)
                        data += tmp
                        while len(tmp) > 0:
                            tmp = f.read(eachCount)
                            data += tmp
                        f.close()
                        md5_data = hashlib.md5(data).hexdigest()
                        send_info = {
                            'type': 'res_model',
                            'size': size,
                            'md5': md5_data,
                            'name': os.path.basename(path)
                        }
                        sock.sendall(b'#info#' + pickle.dumps(send_info))
                        print(sock.recv(1024).decode())
                        sock.send(data)
                        print(sock.recv(1024).decode())
                else:
                    sock.send(b'no')
                r.close()
            elif info['type'] == 'exp':
                size = info['size']
                save_path = args.exp_path + str(info['c_id']) + '_' + str(info['g']) + '/'

                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                check_exist(save_path, info['name'])

                if size > 0:
                    while size > 0:
                        if size >= 204800:
                            d = sock.recv(204800)
                            try:
                                write_file(save_path, info['name'], d)
                            except:
                                print('PermissionError')
                            size -= 204800
                        else:
                            d = sock.recv(size)
                            write_file(save_path, info['name'], d)
                            size = 0

                    sock.send(b'finish')
                else:
                    sock.send(b'file size is 0')
            else:
                raise NotImplementedError()

    sock.send(b'close')
    sock.close()
    print('disconnect socket from', addr)


def write_file(root_path, name, data):
    f = open(root_path + name, 'ab')
    f.write(data)
    f.close()

def check_exist(root_path, name):
    if os.path.exists(root_path + name):
        os.remove(root_path + name)

def generate_model_args(r, g_str):
    agent_list = r.smembers('agent_list')
    winner_list = []
    winner_fitness = []
    all_fitness = []
    new_agent_list = []

    for agent_id in agent_list:
        f_key = 'f_' + agent_id.decode() + '_' + g_str
        if r.hexists('hash', f_key):
            fitness = float(r.hget('hash', f_key))
            all_fitness.append(fitness)
            new_agent_list.append(agent_id)

    f_avg_n = sum(all_fitness)/len(all_fitness)
    for idx, agent_id in enumerate(new_agent_list):
        if all_fitness[idx] >= f_avg_n:
            winner_fitness.append(all_fitness[idx])
            # agent id not always equal to sequence idx
            winner_list.append(int(agent_id))

    if len(all_fitness) == 0:
        new_avg = float(r.hget('hash', 'f_avg'))
    else:
        new_avg = sum(all_fitness)/len(all_fitness)

    model_args = {'len': len(winner_fitness),
                  'actor_path': [],
                  'at_path': [],
                  'ap_path': [],
                  'critic_path': [],
                  'ct_path': [],
                  'fitness': []}
    for winner_id in winner_list:
        p_keys = ['p_' + str(winner_id) + '_' + g_str + '_a',
                  'p_' + str(winner_id) + '_' + g_str + '_at',
                  'p_' + str(winner_id) + '_' + g_str + '_ap',
                  'p_' + str(winner_id) + '_' + g_str + '_c',
                  'p_' + str(winner_id) + '_' + g_str + '_ct']
        actor_path = r.hget('hash', p_keys[0]).decode()
        at_path = r.hget('hash', p_keys[1]).decode()
        ap_path = r.hget('hash', p_keys[2]).decode()
        critic_path = r.hget('hash', p_keys[3]).decode()
        ct_path = r.hget('hash', p_keys[4]).decode()

        model_args['actor_path'].append(actor_path)
        model_args['at_path'].append(at_path)
        model_args['ap_path'].append(ap_path)
        model_args['critic_path'].append(critic_path)
        model_args['ct_path'].append(ct_path)

    for fitness in winner_fitness:
        model_args['fitness'].append(fitness)

    return model_args, new_avg


def generate_next(args, observation_space, action_space, device):
    # use a process to generate next models

    models = generate_models(observation_space, action_space, device)
    models_holder = generate_models(observation_space, action_space, device)

    r = redis.Redis(Settings.redis_host, Settings.redis_port)
    algo = r.hget('hash', 'algo').decode()
    env_name = r.hget('hash', 'env').decode()
    suffix = ''

    while True:
        # periodically update
        time.sleep(Settings.period)

        g = int(r.hget('hash', 'g'))
        n_g = int(r.hget('hash', 'n_g'))

        if g < n_g:
            print('start to perform update for generation', n_g)
            model_args, new_avg = generate_model_args(r, str(g))
            target_path = args.root_path + 'avg_' + str(n_g) + '/'
            target_names = []
            target_names.append(algo + '_actor_' + env_name + '_' + suffix + '_avg_' + str(n_g))
            target_names.append(algo + '_actor_target_' + env_name + '_' + suffix + '_avg_' + str(n_g))
            target_names.append(algo + '_actor_perturbed_' + env_name + '_' + suffix + '_avg_' + str(n_g))
            target_names.append(algo + '_critic_' + env_name + '_' + suffix + '_avg_' + str(n_g))
            target_names.append(algo + '_critic_target_' + env_name + '_' + suffix + '_avg_' + str(n_g))
            # require: fitness, target_path, target_names, len, actor_path,
            #          at_path, ap_path, critic_path, ct_path
            merge_model(models, models_holder, model_args, target_path, target_names)
            r.hset('hash', 'g', str(n_g))
            r.hset('hash', 'f_avg', str(new_avg))

            r.hset('hash', 'p_avg_' + str(n_g) + '_a', target_path + target_names[0])
            r.hset('hash', 'p_avg_' + str(n_g) + '_at', target_path + target_names[1])
            r.hset('hash', 'p_avg_' + str(n_g) + '_ap', target_path + target_names[2])
            r.hset('hash', 'p_avg_' + str(n_g) + '_c', target_path + target_names[3])
            r.hset('hash', 'p_avg_' + str(n_g) + '_ct', target_path + target_names[4])

            print('finish perform update for generation', n_g)

    r.close()


def test_single_merge():
    target_path = 'recv/avg_1/'
    target_names = []
    r = redis.Redis(Settings.redis_host, Settings.redis_port)
    algo = r.hget('hash', 'algo').decode()
    env_name = r.hget('hash', 'env').decode()
    suffix = ''
    n_g = int(r.hget('hash', 'n_g'))
    target_names.append(algo + '_actor_' + env_name + '_' + suffix + '_avg_' + str(n_g))
    target_names.append(algo + '_actor_target_' + env_name + '_' + suffix + '_avg_' + str(n_g))
    target_names.append(algo + '_actor_perturbed_' + env_name + '_' + suffix + '_avg_' + str(n_g))
    target_names.append(algo + '_critic_' + env_name + '_' + suffix + '_avg_' + str(n_g))
    target_names.append(algo + '_critic_target_' + env_name + '_' + suffix + '_avg_' + str(n_g))

    model_args = {'len': 1,
                  'actor_path': ['recv/0_1/dc_actor_Pendulum-v0__0_1'],
                  'at_path': ['recv/0_1/dc_actor_target_Pendulum-v0__0_1'],
                  'ap_path': ['recv/0_1/dc_actor_perturbed_Pendulum-v0__0_1'],
                  'critic_path': ['recv/0_1/dc_critic_Pendulum-v0__0_1'],
                  'ct_path': ['recv/0_1/dc_critic_target_Pendulum-v0__0_1']}

    merge_model(model_args, target_path, target_names)

    r.hset('hash', 'g', str(n_g))
    r.close()

def generate_models(observation_space, action_space, device):
    actor_net_dims = [400, 300]
    actor_net_archs = [F.relu, F.relu]

    critic_net_dims = [400, 300]
    critic_net_archs = [F.relu, F.relu]

    actor = Actor(observation_space, action_space, actor_net_dims, actor_net_archs, 'tanh', None)
    actor_target = Actor(observation_space, action_space, actor_net_dims, actor_net_archs, 'tanh', None)
    actor_perturbed = Actor(observation_space, action_space, actor_net_dims, actor_net_archs, 'tanh', None)

    critic = Critic(observation_space, action_space, critic_net_dims, critic_net_archs, 'mul')
    critic_target = Critic(observation_space, action_space, critic_net_dims, critic_net_archs, 'mul')

    # for param in actor.parameters():
    #     print('actor: ', param.shape)
    #
    # for param in actor_target.parameters():
    #     print('actor_target: ', param.shape)
    #
    # for param in actor_perturbed.parameters():
    #     print('actor_perturbed: ', param.shape)
    #
    # for param in critic.parameters():
    #     print('critic: ', param.shape)
    #
    # for param in critic_target.parameters():
    #     print('critic_target: ', param.shape)

    actor.to(device)
    actor_target.to(device)
    actor_perturbed.to(device)
    critic.to(device)
    critic_target.to(device)

    return [actor, actor_target, actor_perturbed, critic, critic_target]

def test_pair_merge():
    ENV_NAME = 'Pendulum-v0'
    env = utils.makeFilteredEnv(gym.make(ENV_NAME))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    models = generate_models(env, device)
    models_holder = generate_models(env, device)

    model_args = {'len': 3,
                  'actor_path': ['recv/0_0/dc_actor_Pendulum-v0__0_0', 'recv/0_1/dc_actor_Pendulum-v0__0_1',
                                 'recv/0_2/dc_actor_Pendulum-v0__0_2'],
                  'at_path': ['recv/0_0/dc_actor_target_Pendulum-v0__0_0', 'recv/0_1/dc_actor_target_Pendulum-v0__0_1',
                              'recv/0_2/dc_actor_target_Pendulum-v0__0_2'],
                  'ap_path': ['recv/0_0/dc_actor_perturbed_Pendulum-v0__0_0', 'recv/0_1/dc_actor_perturbed_Pendulum-v0__0_1',
                              'recv/0_2/dc_actor_perturbed_Pendulum-v0__0_2'],
                  'critic_path': ['recv/0_0/dc_critic_Pendulum-v0__0_0', 'recv/0_1/dc_critic_Pendulum-v0__0_1',
                                  'recv/0_2/dc_critic_Pendulum-v0__0_2'],
                  'ct_path': ['recv/0_0/dc_critic_target_Pendulum-v0__0_0', 'recv/0_1/dc_critic_target_Pendulum-v0__0_1',
                              'recv/0_2/dc_critic_target_Pendulum-v0__0_2'],
                  'fitness': [-571.5893259546909, -821.34682349, -569.4976427318661]}


    r = redis.Redis(Settings.redis_host, Settings.redis_port)
    algo = r.hget('hash', 'algo').decode()
    env_name = r.hget('hash', 'env').decode()
    suffix = ''
    n_g = int(r.hget('hash', 'n_g'))

    target_path = 'recv/avg_' + str(n_g) + '/'
    target_names = []

    target_names.append(algo + '_actor_' + env_name + '_' + suffix + '_avg_' + str(n_g))
    target_names.append(algo + '_actor_target_' + env_name + '_' + suffix + '_avg_' + str(n_g))
    target_names.append(algo + '_actor_perturbed_' + env_name + '_' + suffix + '_avg_' + str(n_g))
    target_names.append(algo + '_critic_' + env_name + '_' + suffix + '_avg_' + str(n_g))
    target_names.append(algo + '_critic_target_' + env_name + '_' + suffix + '_avg_' + str(n_g))

    merge_model(models, models_holder, model_args, target_path, target_names)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='recv/')
    parser.add_argument('--exp_path', type=str, default='recv/exps/')
    parser.add_argument('--env_name', type=str, default='')
    parser.add_argument('--mode', type=str, default='next')  # server/next

    args = parser.parse_args()

    args.env_name = 'MyEnv-10'

    if args.mode == 'next':
        # env = utils.makeFilteredEnv(gym.make(args.env_name))
        n_users = int(args.env_name[-2:])
        observation_space, action_space = get_space(n_users * 7 + 3, n_users * 2)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        generate_next(args, observation_space, action_space, device)
    else:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((Settings.master_host, Settings.master_port))
        s.listen(5)

        # create folder to store files
        if not os.path.exists(args.root_path):
            os.makedirs(args.root_path)

        print('Waiting for connection...')
        while True:
            sock, addr = s.accept()
            # t = threading.Thread(target=tcplink(sock, addr, args))
            try:
                _thread.start_new_thread(tcplink, (sock, addr, args))
            except:
                print('Error: unable to start thread')
