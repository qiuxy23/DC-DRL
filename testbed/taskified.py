import torch
import cv2
import face_recognition
import time


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

def face_recognition_in_known_faces(input_video_path, return_dict):
    print('perform offloading tasks: face_recognition_in_known_faces')

    input_movie = cv2.VideoCapture(input_video_path)

    res = {'video_path: ': input_video_path,
           'data': []}

    edge_cpu_begin = time.time()
    frame_number = 0

    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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
            return_dict[frame_number] = face_names
    return_dict['edge_cpu_time'] = int(time.time() - edge_cpu_begin)
    return res


'''
def face_recognition_in_known_faces(frames, left, right, return_dict):

    print('perform offloading tasks: face_recognition_in_known_faces in range {} and {}'.format(left, right))

    res = []

    frame_number = left
    for frame in frames:
        frame_number += 1

        rgb_frame = frame[:, :, ::-1]

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
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
            return_dict[frame_number] = face_names
            res.append((frame_number, face_names))
    return res
'''

# Export functions as tasks
# in this testbed, we only consider face recognition tasks
# future work can consider more tasks
tasks = [face_recognition_in_known_faces]