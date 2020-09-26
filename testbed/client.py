import torch
import argparse
import numpy as np
import os
import random
import socket
import time
import face_recognition
import cv2
import pickle
import threading

def offload_to_edge(input_video_path, client_id, SERVER_HOST, SERVER_PORT):
    begin = time.time()
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((SERVER_HOST, SERVER_PORT))

    info = b'#info#'
    video_size = os.path.getsize(input_video_path)
    offload_args = {
        'client_id': client_id,
        'task_id': 0,
        'file_name': input_video_path,
        'video_size': video_size
    }
    info += pickle.dumps(offload_args)

    video = open(input_video_path, 'rb')
    data = b''

    eachCount = 4096
    tmp = video.read(eachCount)
    data += tmp
    while len(tmp) > 0:
        tmp = video.read(eachCount)
        data += tmp
    video.close()

    client_socket.sendall(info)
    print(client_socket.recv(1024).decode())
    client_socket.sendall(data)
    print(client_socket.recv(1024).decode())
    res_info = pickle.loads(client_socket.recv(1024))
    print('res_info:', res_info)
    res_length = res_info['res_length']

    client_socket.send(b'receive res info')
    res = b''
    while res_length > 0:
        data = client_socket.recv(2048)
        if len(data) == 10240:
            res_length -= 10240
        else:
            res_length -= len(data)
        res += data
    res = pickle.loads(res)
    print('res:', res)

    print('Process video at edge {} with [{}] seconds.'.format(input_video_path, int(time.time() - begin)))
    print('Occupy edge cpu time with [{}] seconds.'.format(int(res['edge_cpu_time'])))



def process_local(known_faces, input_video_path, pool_sema):
    begin = time.time()
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    with pool_sema:
        cpu_begin = time.time()

        input_movie = cv2.VideoCapture(input_video_path)

        res = {'video_path: ': input_video_path,
               'data': []}

        frame_number = 0
        while True:
            # Grab a single frame of video
            ret, frame = input_movie.read()
            frame_number += 1

            # Quit when the input video file ends
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5
            )

            rgb_frame = frame[:, :, ::-1]

            # Find all the faces and face encodings in the current frame of video
            # face_locations = face_recognition.face_locations(rgb_frame)
            # face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            face_encodings = face_recognition.face_encodings(rgb_frame, faces)

            for f_i in range(len(faces)):
                left, top, size = faces[f_i][0], faces[f_i][1], faces[f_i][2]
                bottm = top + size
                right = left + size
                faces[f_i] = [top, right, bottm, left]

            face_names = []

            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)

                # If you had more than 2 faces, you could make this logic a lot prettier
                # but I kept it simple for the demo
                name = None
                if match[0]:
                    name = 'Ben Affleck (Batman)'
                elif match[1]:
                    name = 'Gal Gadot (Wonder Woman)'
                elif match[2]:
                    name = 'Henry Cavill (Superman)'

                if name:
                    face_names.append(name)

            # Write the resulting image to the output video file
            if len(face_names) > 0:
                res['data'].append((frame_number, face_names))

        print('res', res)
        print('Process video locally {} with [{}] seconds.'.format(input_video_path, int(time.time() - begin)))
        print('Occupy local cpu time with [{}] seconds.'.format(int(time.time() - cpu_begin)))



def run_client():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server_host', type=str, default='localhost')
    parser.add_argument('--server_port', type=int, default=10086)
    parser.add_argument('--client_id', type=int, default=1)
    parser.add_argument('--load_path_prefix', type=str, default='actions_')
    parser.add_argument('--video_folder', type=str, default='videos/')
    parser.add_argument('--conf')

    args = parser.parse_args()
    load_path = '{}{}{}'.format(args.load_path_prefix, str(args.client_id), '.npy')

    actions = np.load(load_path)
    # print(actions)

    input_video_paths = sorted(os.listdir(args.video_folder))
    input_video_paths = [args.video_folder + path for path in input_video_paths]

    # for path in input_video_paths:
    #     print(path)


    SERVER_HOST = args.server_host
    SERVER_PORT = args.server_port

    ben_image = face_recognition.load_image_file('images/Ben_Affleck.jpg')
    ben_face_encoding = face_recognition.face_encodings(ben_image)[0]

    gal_image = face_recognition.load_image_file('images/Gal_Gadot.jpg')
    gal_face_encoding = face_recognition.face_encodings(gal_image)[0]

    hen_image = face_recognition.load_image_file('images/Henry_Cavill.jpg')
    hen_face_encoding = face_recognition.face_encodings(hen_image)[0]

    known_faces = [
        ben_face_encoding,
        gal_face_encoding,
        hen_face_encoding
    ]

    total_begin = time.time()
    thread_list = []

    # for FIFO
    max_connections = 1
    pool_sema = threading.BoundedSemaphore(max_connections)

    for input_video_path, action in zip(input_video_paths, actions):
        # to model the epoch length
        if action:
            t = threading.Thread(target=offload_to_edge,
                                 args=[input_video_path, args.client_id, SERVER_HOST, SERVER_PORT], daemon=True)
            t.start()
            thread_list.append(t)
        else:
            t = threading.Thread(target=process_local,
                                 args=[known_faces, input_video_path, pool_sema], daemon=True)
            t.start()
            thread_list.append(t)
        time.sleep(10)
    for t in thread_list:
        t.join()
    print('Complete all tasks with {} seconds.'.format(int(time.time() - total_begin)))


if __name__ == '__main__':
    # arr = []
    # for i in range(23):
    #     arr.append(0)
    # print(arr)
    # np.save('actions_1.npy', arr)
    # exit(1)

    # [0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0]
    run_client()
