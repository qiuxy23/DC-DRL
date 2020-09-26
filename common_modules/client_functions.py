# -*- coding: utf-8 -*-
import socket
import argparse
import os
import hashlib
import pickle

from conf.default import Settings
from common_modules.parameter_servers import write_file
from common_modules.parameter_servers import check_exist
 
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', help='the file you want to send')
    parser.add_argument('-i', '--ip', help='target ip address')
    args = parser.parse_args()
    path = args.file
    ip = args.ip
    print('File path :', path)
    return path, ip
 
def read_file(path, eachCount):
    f = open(path, 'rb')
    size = os.path.getsize(path)
    print('Sent file size:', size)
    data = b''
    tmp = f.read(eachCount)
    data += tmp
    while len(tmp) >0 :
        tmp = f.read(eachCount)
        data += tmp
    f.close()
    md5_data = hashlib.md5(data).hexdigest()
    info = {
        'size': size,
        'md5': md5_data,
        'name': os.path.basename(path)
    }
    return data, info


def send_model_c(path, fitness, c_id, g, model_type, host, port):
    d, info = read_file(path, 204800)

    info['type'] = 'model'
    info['f'] = fitness
    info['c_id'] = c_id
    info['g'] = g
    info['mt'] = model_type

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    print(s.recv(1024).decode())
    s.sendall(b'#info#' + pickle.dumps(info))
    print(s.recv(1024).decode())
    s.send(d)
    print(s.recv(1024).decode())
    s.send(b'exit')
    print(s.recv(1024).decode())
    s.close()

def join_population(c_id, host, port):
    info = {'type': 'join',
            'c_id': c_id}
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    print(s.recv(1024).decode())
    s.sendall(b'#info#' + pickle.dumps(info))
    print(s.recv(1024).decode())
    s.send(b'exit')
    print(s.recv(1024).decode())
    s.close()

def exit_population(c_id, host, port):
    info = {'type': 'exit',
            'c_id': c_id}
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    print(s.recv(1024).decode())
    s.sendall(b'#info#' + pickle.dumps(info))
    print(s.recv(1024).decode())
    s.send(b'exit')
    print(s.recv(1024).decode())
    s.close()

def request_model(c_id, fitness, host, port, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    info = {'type': 'request',
            'c_id': c_id,
            'f': fitness}
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    print(s.recv(1024).decode())
    s.sendall(b'#info#' + pickle.dumps(info))
    print(s.recv(1024).decode())
    msg = s.recv(1024).decode()
    print(msg)
    res_paths = []
    if msg == 'yes':
        for i in range(5):
            data = s.recv(2048)
            if data.startswith(b'#info#'):
                str_info = data[6:]
                res_info = pickle.loads(str_info)
                print(res_info)
                size = res_info['size']
                s.send(b'ack info')
                check_exist(save_path, res_info['name'])
                if size > 0:
                    while size > 0:
                        if size >= 4096:
                            d = s.recv(4096)
                            write_file(save_path, res_info['name'], d)
                            size -= 4096
                        else:
                            d = s.recv(size)
                            write_file(save_path, res_info['name'], d)
                            size = 0
                s.send(b'finish')
                res_paths.append(save_path + res_info['name'])
        s.send(b'exit')
        print(s.recv(1024).decode())
        s.close()
    else:
        s.send(b'exit')
        print(s.recv(1024).decode())
        s.close()
    return res_paths

def send_exp_c(c_id, g, path, host, port):

    d, info = read_file(path, 204800)
    info['type'] = 'exp'
    info['c_id'] = c_id
    info['g'] = g

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    print(s.recv(1024).decode())
    s.sendall(b'#info#' + pickle.dumps(info))
    print(s.recv(1024).decode())
    s.send(d)
    print(s.recv(1024).decode())
    s.send(b'exit')
    print(s.recv(1024).decode())
    s.close()


if __name__ == '__main__':
    # cands = ['dc_actor_Pendulum-v0__0_0', 'dc_actor_perturbed_Pendulum-v0__0_0', 'dc_actor_target_Pendulum-v0__0_0',
    #          'dc_critic_Pendulum-v0__0_0', 'dc_critic_target_Pendulum-v0__0_0']
    # path = 'models/0_0/' + cands[0]
    # fitness = -571.5893259546909
    # # fitness = -813.4483744207146
    # # a, at, ap, c, ct
    # model_type = 'a'
    # c_id = 0
    # g = 0
    #
    # # send_model_c(path, fitness, c_id, g, model_type, Settings.master_host, Settings.master_port)

    # join_population(12, Settings.master_host, Settings.master_port)
    # exit_population(12, Settings.master_host, Settings.master_port)

    # c_id = 1
    # fitness = -1
    # request_model(c_id, fitness, Settings.master_host, Settings.master_port, str(c_id) + '_temp/')

    c_id = 1
    g = 0
    path = 'exps/exp_1_1_10000.npy'
    send_exp_c(c_id, g, path, Settings.master_host, Settings.master_port)
