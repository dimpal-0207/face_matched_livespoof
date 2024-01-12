import dlib

# Load a pre-trained face landmark model
predictor = dlib.shape_predictor("model_dlib/shape_predictor_68_face_landmarks.dat")

#final project app for realtime face detection and antispoofing frontend into index.html
import base64
import datetime
import logging
import os
import time

from pytz import timezone


import cv2
import face_recognition
import numpy as np
import requests
from flask import Flask, render_template, request
from flask_cors import CORS
from flask_socketio import SocketIO


# Import the 'test' function from your existing code
from test import test
indian_timezone = timezone('Asia/Kolkata')

# Configure logging
logging.basicConfig(level=logging.INFO, filename="file.log", filemode="a", format="%(asctime)s%(levelname)s:%(name)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S %Z" ) # Customize the date format he)
logger = logging.getLogger()  # G
list_connected_sockets = []

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, ping_interval=10000, ping_timeout=5000, reconnection=True, cors_allowed_origins="*", cookie=False)


video_capture = cv2.VideoCapture(0)

@app.route('/')
def index():
    return render_template("index.html")


image_database = {}


@socketio.on('connect')
def handle_connect():
    logging.info("Client connected")
    socketio.emit('connect', {'status': 'connected with server'})

import dlib

@socketio.on('get_face')
def face_data(data):
    print("====data", len(data))
    try:
        if data:
            binary_data = base64.b64decode(data)
            # print("===>binary", binary_data)
            image_np = cv2.imdecode(np.frombuffer(binary_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image_np is not None:
                face_encodings = face_recognition.face_encodings(image_np)
                if face_encodings:
                    # Assuming you want to emit for each detected face
                    for face_encoding in face_encodings:
                        # print("===>encode", len(face_encoding))
                        logging.info('encode len  of facedata: %s', len(face_encoding))
                        socketio.emit('get_face', {"status": 200, 'message': 'Face Detected!'})
                else:
                    logging.info('encode len  of facedata: %s', "Not get face encodings")
                    socketio.emit('get_face', {"status": 400, 'message': 'Please upload a proper face image'})
        else:
            socketio.emit('get_face', {"status": 400,'message': 'Not received face data'})
    except Exception as e:
        logging.error(f"Error: {e}")



@socketio.on('join_Room')
def joinroom(data):
    logging.info("userid_data: %s", data)
    # print("===data", data)
    global image_database
    try:
        # Hit the API with the userid
        api_url = f"http://13.126.129.218:6002/api/auth/user-profile-image/{data}"
        response = requests.get(api_url)
        # print("===response", response)
        logging.info('response: %s', response)
        # Check the API response and perform actions accordingly
        if response.status_code == 200:
            api_data = response.json()
            # print("===>api_data", api_data)# Assuming the API returns JSON data
            profile_image_data = api_data.get('data', [])[0].get('profileImage', None)
            face_name = api_data.get('data', [])[0].get('profileImagePath', None).split('/')[-1].split('.')[0]
            # print("===profile_image_path", face_name)
            # print("===profile_image_data", len(profile_image_data))
            binary_data = base64.b64decode(profile_image_data)
            image_np = cv2.imdecode(np.frombuffer(binary_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            face_encoding = face_recognition.face_encodings(image_np)[0]
            logging.info('faceencoding of api face: %s', len(face_encoding))

            # print("===faceencoding of api face", face_encoding)
            # face_name = "profileImage"
            image_database = {face_name : face_encoding}
            return image_database
    except Exception as e:
        logging.error(f"Error: {e}")


@socketio.on('disconnect')
def handle_disconnect():
    logging.info("Client disconnected")
    socketio.emit('disconnect_response', {'status': 'disconnected'})

# Event handler for streaming frames from the client


face_recognition_model = dlib.get_frontal_face_detector()


def align_face(image, face_location):
    rect = dlib.rectangle(face_location[3], face_location[0], face_location[1], face_location[2])
    shape = predictor(image, rect)
    desired_left_eye = (0.35, 0.35)
    desired_right_eye = (0.65, 0.35)
    d_x = shape.part(1).x - shape.part(0).x
    d_y = shape.part(1).y - shape.part(0).y
    angle = np.degrees(np.arctan2(d_y, d_x)) - 180
    desired_dist = (desired_right_eye[0] - desired_left_eye[0], desired_right_eye[1] - desired_left_eye[1])
    desired_dist = np.linalg.norm(desired_dist)
    scale = desired_dist / np.linalg.norm((d_x, d_y))
    M = cv2.getRotationMatrix2D((shape.part(2).x, shape.part(2).y), angle, scale)
    aligned_face = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return aligned_face


# working=====>
@socketio.on('stream_frame')
def handle_webcam_frame(data):
    # output_directory = "output_directory"
    # timestamp = datetime.datetime.now()
    #
    # if not os.path.exists(output_directory):
    #     os.makedirs(output_directory)
    # timestamp_str = timestamp.strftime("%Y-%m-%d_%H-%M-%S.%f")[:-3]  # Format the timestamp
    #
    # file_path = f'output_directory/file1_{timestamp_str}.png'
    # with open(file_path, "wb") as f:
    #     f.write(base64.b64decode(data))
    try:
        frame_data = data.split(',')[1]
        # Decode the base64 encoded image
        binary_data = base64.b64decode(frame_data)
        frame_np = cv2.imdecode(np.frombuffer(binary_data, dtype=np.uint8), cv2.IMREAD_COLOR)

        # Check if 'frame_np' is not None before using it
        if frame_np is not None:
            # Perform anti-spoofing test using the 'test' function
            label, confidence = test(image=frame_np, model_dir=os.path.join("resources", "anti_spoof_models"),
                                     device_id=0)
            print("===lable", label)
            spoofing_threshold = 0.5
            if label == 1:
                # Proceed with face recognition if not spoofed
                rgb_frame = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition_model(frame_np, 2)

                print("==face_locations", face_locations)
                if len(face_locations) > 0:
                    # Extract face encodings for all detected faces
                    for face_location in face_locations:
                        top, right, bottom, left = face_location.top(), face_location.right(), face_location.bottom(), face_location.left()
                        face_encodings = face_recognition.face_encodings(frame_np, [(top, right, bottom, left)])
                        print("==face_encoding", face_encodings)
                        # Initialize result dictionary
                        # Check for face match with images in the image_database
                        for face_encoding in face_encodings:
                            matches = face_recognition.compare_faces(list(image_database.values()), face_encoding)
                            print("=====>matches", matches)
                            logging.info("match %s", matches)
                            if any(matches):
                                name = list(image_database.keys())[matches.index(True)]
                                result = {'matched': True, 'name': name, 'message': 'Match Found!'}
                                image_database.clear()
                                logging.info("=====imagedatabase %s:", len(image_database))
                                socketio.emit('face_recognition_result', result)
                                break
                            else:
                                image_database.clear()
                                logging.info("match not with db=====imagedatabase %s:", len(image_database))
                                socketio.emit('face_recognition_result',
                                              {'matched': False, 'name': "Unknown", 'message': 'not matching face with the DB stored image'})

            else:
                image_database.clear()
                logging.info("spoof detect=====imagedatabase %s:", len(image_database))
                socketio.emit('face_recognition_result', {'matched': False, 'message': 'Please provide a real face'})
        else:
            image_database.clear()
            logging.info("no frame=====imagedatabase %s:", len(image_database))
            socketio.emit('face_recognition_result', {'matched': False, 'message': 'Failed to decode the frame'})
    except Exception as e:
        socketio.emit('error', {'message': str(e)})
        logging.error(f"Error: {e}")


# @socketio.on('stream_frame')
# def process_stream_frame(data):
#     # ... (previous code)
#     try:
#         data = data.split(',')[1]
#         binary_data = base64.b64decode(data)
#         frame_np = cv2.imdecode(np.frombuffer(binary_data, dtype=np.uint8), cv2.IMREAD_COLOR)
#
#         if frame_np is not None:
#             # Detect faces in the frame
#             face_locations = face_recognition.face_locations(frame_np)
#
#             if face_locations:
#                 # Use the first detected face for simplicity (you can iterate over all faces)
#                 top, right, bottom, left = face_locations[0]
#
#                 # Use dlib to find facial landmarks
#                 shape = predictor(frame_np, dlib.rectangle(left=int(left), top=int(top), right=int(right), bottom=int(bottom)))
#
#                 # Get facial landmarks using the parts() method
#                 landmarks = shape.parts()
#
#                 # Align face based on facial landmarks (you may need to implement this based on your requirements)
#                 aligned_face = align_face(frame_np, landmarks)
#
#                 # Encode the aligned face
#                 live_face_encoding = face_recognition.face_encodings(aligned_face)[0]
#
#                 # Compare face encodings with the image database
#                 matches = face_recognition.compare_faces(list(image_database.values()), live_face_encoding)
#
#                 # Check if there is a match
#                 if any(matches):
#                     name = list(image_database.keys())[matches.index(True)]
#                     result = {'matched': True, 'name': name, 'message': 'Match Found!'}
#                     socketio.emit('face_recognition_result', result)
#                     logging.info("Name: %s", result)
#                 else:
#                     socketio.emit('face_recognition_result',
#                                   {'matched': False, 'name': "Unknown", 'message': 'No match found'})
#             else:
#                 socketio.emit('face_recognition_result',
#                               {'matched': False, 'name': "Unknown", 'message': 'No face detected'})
#         else:
#             socketio.emit('face_recognition_result', {'matched': False, 'name': "Unknown",
#                                                       'message': 'Failed to decode the frame'})
#     except Exception as e:
#         socketio.emit('error', {'message': str(e)})
#         logging.error(f"Error: {e}")


if __name__ == '__main__':
    socketio.run(app, host="0.0.0.0", debug=True, port=5001)
    video_capture.release()
    cv2.destroyAllWindows()
