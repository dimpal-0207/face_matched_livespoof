#final project app for realtime face detection and antispoofing frontend into index.html

import base64
import datetime
import logging
import os

import boto3
import botocore
import cv2
import face_recognition
import numpy as np
import requests
from flask import Flask, render_template, request
from flask_cors import CORS
from flask_socketio import SocketIO
from decouple import config
import time

# Import the 'test' function from your existing code
from test import test

# Configure logging
logging.basicConfig(level=logging.INFO, filename="file.log", filemode="a", format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger()  # G
list_connected_sockets = []

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, ping_interval=10000, ping_timeout=5000, reconnection=True, cors_allowed_origins="*", cookie=False)


video_capture = cv2.VideoCapture(0)


# Decode base64 image and create a face encoding for the image
def decode_and_encode_base64_image(base64_data):
    binary_data = base64.b64decode(base64_data)
    image_np = cv2.imdecode(np.frombuffer(binary_data, dtype=np.uint8), cv2.IMREAD_COLOR)
    face_encoding = face_recognition.face_encodings(image_np)
    return face_encoding[0] if face_encoding else None


@app.route('/')
def index():
    return "<h3>Welcome to Facedetection App</h3>"

# Event handler for client connection


image_database = {}


@socketio.on('connect')
def handle_connect():
    logging.info("Client connected")
    socketio.emit('connect', {'status': 'connected with server'})


@socketio.on('join_Room')
def joinroom(data):
    logging.info("userid_data: %s", data)
    print("===data", data)
    global image_database
    try:
        # Hit the API with the userid
        # api_url = f"http://192.168.1.6:5007/api/auth/user-profile-image/{data}" #for local
        api_url = f"http://3.111.167.55:6001/api/auth/user-profile-image/{data}" # for live api url
        # payload = {'user_id': user_id}
        response = requests.get(api_url)
        print("===response", response)
        # Check the API response and perform actions accordingly
        if response.status_code == 200:
            api_data = response.json()  # Assuming the API returns JSON data
            profile_image_data = api_data.get('data', [])[0].get('profileImage', None)
            binary_data = base64.b64decode(profile_image_data)
            image_np = cv2.imdecode(np.frombuffer(binary_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            face_encoding = face_recognition.face_encodings(image_np)[0]
            # print("===>face_encode", face_encoding)
            face_name = "profileImage"
            image_database = {face_name : face_encoding}
            return image_database
    except Exception as e:
        logging.error(f"Error: {e}")


@socketio.on('disconnect')
def handle_disconnect():
    logging.info("Client disconnected")
    socketio.emit('disconnect_response', {'status': 'disconnected'})


# Event handler for streaming frames from the client
@socketio.on('stream_frame')
def handle_webcam_frame(data):
    try:
        # Decode the base64 encoded image
        frame_data = data
        binary_data = base64.b64decode(frame_data)
        frame_np = None  # Set a default value before the 'try' block
        frameArray = np.frombuffer(binary_data, dtype=np.uint8)

        try:
            frame_np = cv2.imdecode(frameArray, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"Error decoding frame: {e}")

        # Get the directory of the current script
        script_directory = os.path.dirname(os.path.abspath(__file__))
        # Construct the dynamic path relative to the script's location
        resources_directory = os.path.join(script_directory, "resources", "anti_spoof_models")

        # Check if 'frame_np' is not None before using it
        if frame_np is not None:
            # Perform anti-spoofing test using the 'test' function
            label, confidence = test(image=frame_np, model_dir=resources_directory, device_id=0)

            spoofing_threshold = 0.5

            if label == 1 and confidence > spoofing_threshold:
                # Proceed with face recognition if not spoofed
                rgb_frame = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)

                # Detect faces in the frame
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                # Initialize result dictionary
                result = {'matched': False, 'message': 'not matching face with the DB stored image'}

                for face_encoding in face_encodings:
                    # Check for face match with images in the image_database
                    matches = face_recognition.compare_faces(list(image_database.values()), face_encoding)
                    # If a match is found, update the result dictionary
                    if True in matches:
                        first_match_index = matches.index(True)
                        name = list(image_database.keys())[first_match_index]
                        result = {'matched': True,  'message': 'Match Found!'}
                        logging.info("Name: %s", result)
                        break

                # Emit the results to the connected client
                socketio.emit('face_recognition_result', result)
            else:
                socketio.emit('face_recognition_result', {'matched': False,'message': 'please provide real face'})
                logging.error("please provide real face")
        else:
            # Emit an appropriate response to the client
            socketio.emit('face_recognition_result', {'matched': False,'message': 'failed to detect the face'})
            logging.error("frame does not detect a proper face")

    except Exception as e:
        # Emit an error response to the client
        socketio.emit('error', {'message': str(e)})
        logging.error(f"Error: {e}")


# Main entry point


if __name__ == '__main__':
    socketio.run(app, host="0.0.0.0" ,debug=True, port=5001)
    video_capture.release()
    cv2.destroyAllWindows()






























