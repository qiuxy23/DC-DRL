import face_recognition
import cv2
import numpy as np
import pickle

# This is a demo of running face recognition on a video file and saving the results to a new video file.
#
# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Open the input movie file
input_movie = cv2.VideoCapture('hamilton_clip.mp4')
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

# Load some sample pictures and learn how to recognize them.
lmm_image = face_recognition.load_image_file('lin-manuel-miranda.png')
lmm_face_encoding = face_recognition.face_encodings(lmm_image)[0]

al_image = face_recognition.load_image_file('alex-lacamoire.png')
al_face_encoding = face_recognition.face_encodings(al_image)[0]

known_faces = [
    lmm_face_encoding,
    al_face_encoding
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
frame_number = 0

while True:
    # Grab a single frame of video
    ret, frame = input_movie.read()

    frame_number += 1
    # with open('frame.pickle', 'wb') as f:
    #     pickle.dump(frame, f)

    # with open('frame.pickle', 'rb') as f:
    #     frame = pickle.load(f)

    # Quit when the input video file ends
    if not ret:
        break

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
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
            name = 'Lin-Manuel Miranda'
        elif match[1]:
            name = 'Alex Lacamoire'

        face_names.append(name)

    # Write the resulting image to the output video file
    print('Processing frame {} / {}, found person: {}'.format(frame_number, length, face_names))

# All done!
input_movie.release()
cv2.destroyAllWindows()
