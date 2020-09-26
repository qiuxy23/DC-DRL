import torch
import face_recognition
import pickle
import cv2
import time


# Load some sample pictures and learn how to recognize them.
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

input_movies = ['videos/Batman_vs_Superman_Official_Teaser_Trailer_cut_1.mp4',
                'videos/Batman_vs_Superman_Official_Teaser_Trailer_cut_2.mp4',
                'videos/Batman_vs_Superman_Official_Teaser_Trailer_cut_3.mp4',
                'videos/Batman_vs_Superman_Official_Trailer_cut_1.mp4',
                'videos/Batman_vs_Superman_Official_Trailer_cut_2.mp4',
                'videos/Batman_vs_Superman_Official_Trailer_cut_3.mp4',
                'videos/Batman_vs_Superman_Official_Trailer_cut_4.mp4',
                'videos/Batman_vs_Superman_Official_Trailer_cut_5.mp4',
                'videos/Batman_vs_Superman_Official_Trailer_cut_6.mp4',
                'videos/Justice_League_Official_Trailer_1_cut_1.mp4',
                'videos/Justice_League_Official_Trailer_1_cut_2.mp4',
                'videos/Justice_League_Official_Trailer_1_cut_3.mp4',
                'videos/Justice_League_Official_Trailer_1_cut_4.mp4',
                'videos/Justice_League_Official_Trailer_1_cut_5.mp4',
                'videos/Zack_Snyder_Justice_League_Official_Teaser_1_cut_1.mp4',
                'videos/Zack_Snyder_Justice_League_Official_Teaser_1_cut_2.mp4',
                'videos/Zack_Snyder_Justice_League_Official_Teaser_1_cut_3.mp4',
                'videos/Zack_Snyder_Justice_League_Official_Teaser_1_cut_4.mp4',
                'videos/Zack_Snyder_Justice_League_Official_Teaser_1_cut_5.mp4',
                'videos/Zack_Snyder_Justice_League_Official_Teaser_1.mp4',
                'videos/Zack_Snyder_Justice_League_Sneak_Peak_2.mp4',
                'videos/Zack_Snyder_Justice_League_Sneak_Peak_3.mp4',
                'videos/Zack_Snyder_Justice_League_Sneak_Peak.mp4']


for path in input_movies:
    begin = time.time()
    input_movie = cv2.VideoCapture(path)

    length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    frame_number = 0

    while True:
        # Grab a single frame of video
        ret, frame = input_movie.read()

        frame_number += 1

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
                name = 'Ben Affleck'
            elif match[1]:
                name = 'Gal Gadot'
            elif match[2]:
                name = 'Henry Cavill'

            if name:
                face_names.append(name)

        # Write the resulting image to the output video file
        if len(face_names) > 0:
            print('Processing frame {} / {} for {}, found person: {}'.format(frame_number, length, path[7:], face_names))

    end = time.time()
    print('Processing time {} for {}'.format(end - begin, path[7:]))
