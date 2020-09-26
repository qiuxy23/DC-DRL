import torch
import pickle
import socket
import struct
import threading
import time
import random
import psutil
import cv2
import multiprocessing
from multiprocessing import Manager

from testbed.taskified import tasks

def init_task_names():
    return [task.__name__ for task in tasks]


class MyThread(threading.Thread):
    def __init__(self, func, args):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None


def write_data(path, data):
    f = open(path, 'ab')
    f.write(data)
    f.close()

def on_new_client(conn, pool_sema, cpus):
    with pool_sema:
        try:
            while True:
                data = conn.recv(2048)
                if not data.startswith(b'#info#'):
                    print(data)
                if data == b'close':
                    conn.close()
                    break
                elif data.startswith(b'#info#'):
                    info = pickle.loads(data[6:])
                    print('info:', info)
                    conn.send(b'receive info')
                    size = info['video_size']
                    total_data = b''
                    while size > 0:
                        data = conn.recv(10240)
                        if len(data) == 10240:
                            size -= 10240
                        else:
                            size -= len(data)
                        total_data += data
                    conn.send(b'receive data')
                    write_data('recv/' + info['file_name'], total_data)


                    spec_task = tasks[info['task_id']]
                    input_video_path = 'recv/' + info['file_name']

                    res = {'video_path: ': input_video_path,
                           'data': [],
                           'edge_cpu_time': 0}

                    manager = Manager()
                    return_dict = manager.dict()
                    p = multiprocessing.Process(target=spec_task, args=(input_video_path, return_dict))
                    p.start()
                    p.join()


                    '''
                    # a multi-process implement for each tasks
                    # due to the tasks is small, we abandon this implement 
                    input_movie = cv2.VideoCapture(input_video_path)

                    res = {'video_path: ': input_video_path,
                           'data': []}

                    frames = []
                    while True:
                        ret, frame = input_movie.read()
                        if not ret:
                            break
                        frames.append(frame)

                    left = 0
                    frame_step = int(len(frames)/cpus)
                    process_list = []
                    for i_cpu in range(cpus):
                        p = multiprocessing.Process(target=spec_task,
                                                    args=(frames[left: left + frame_step], left, left + frame_step, return_dict))
                        p.start()
                        process_list.append(p)
                        left += frame_step

                    for p in process_list:
                        p.join()
                        
                        
                    # for t in threads_list:
                    #     res = t.get_result()
                    #     for val in res:
                    #         res['data'].append(val)
                    '''


                    for key in return_dict.keys():
                        if key is not 'edge_cpu_time':
                            res['data'].append((key, return_dict[key]))
                    res['edge_cpu_time'] = return_dict['edge_cpu_time']
                    # res = spec_task('recv/' + info['file_name'])
                    res = pickle.dumps(res)
                    res_info = {
                        'res_length': len(res)
                    }
                    print('res_info in server:', res_info)
                    conn.sendall(pickle.dumps(res_info))
                    print(conn.recv(1024).decode())
                    conn.sendall(res)
                else:
                    raise NotImplementedError()
        except ConnectionError:
            # Client disconnected
            print('Client disconnected')
            conn.close()
        except NotImplementedError:
            print('NotImplementedError')
            conn.close()
        except Exception as e:
            print('Unknown exception:', e)
            conn.close()
        conn.close()



def run_server(HOST, PORT):

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print('Socket created')

    s.bind((HOST, PORT))
    print('Socket bind complete')
    s.listen(10)
    print('Socket now listening on port', PORT)

    print('Waiting for client to connect')

    task_names = init_task_names()
    # for FIFO
    max_connections = 3
    pool_sema = threading.BoundedSemaphore(max_connections)
    # should change with sema
    cpus = psutil.cpu_count()

    while True:

        # Receive connection from client
        client_socket, (client_ip, client_port) = s.accept()
        print('Received connection from:', client_ip, client_port)

        # Start a new thread for the client. Use daemon threads to make exiting the server easier
        # Set a unique name to display all images
        t = threading.Thread(target=on_new_client, args=[client_socket, pool_sema, cpus], daemon=True)
        t.setName(str(client_ip) + ':' + str(client_port))
        t.start()
        print('Started thread with name:', t.getName())


if __name__ == '__main__':
    HOST = 'localhost'
    # HOST = '192.168.199.197'
    PORT = 10086

    run_server(HOST, PORT)

# sudo nano /etc/wpa_supplicant/wpa_supplicant.conf
# ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
# update_config=1
# network={
#   ssid="wifi"
#   psk="12331233"
#   key_mgmt=WPA-PSK
# }

#
# network={
# 	ssid="ssid"
# 	key_mgmt=WPA-EAP IEEE8021X
# 	eap=PEAP
# 	identity="username"
# 	password="password"
# }
