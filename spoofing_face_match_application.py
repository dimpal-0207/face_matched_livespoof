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
    return "<h3>Welcome to Facedetection App</h3>"


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


# working=====>
@socketio.on('stream_frame')
def handle_webcam_frame(data):
    output_directory = "output_directory"
    timestamp = datetime.datetime.now()

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    timestamp_str = timestamp.strftime("%Y-%m-%d_%H-%M-%S.%f")[:-3]  # Format the timestamp

    file_path = f'output_directory/file1_{timestamp_str}.png'
    with open(file_path, "wb") as f:
        f.write(base64.b64decode(data))
    try:
        # Decode the base64 encoded image
        binary_data = base64.b64decode(data)
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
                face_locations = face_recognition_model(frame_np, 1)

                print("==face_locations", face_locations)
                if len(face_locations) > 0:
                    # Extract face encodings for all detected faces
                    for face_location in face_locations:
                        top, right, bottom, left = face_location.top(), face_location.right(), face_location.bottom(), face_location.left()
                        face_encodings = face_recognition.face_encodings(frame_np, [(top, right, bottom, left)])
                        print("==face_encoding", face_encodings)

                        # Initialize result dictionary
                        result = {'matched': False, 'name': "Unknown", 'message': 'not matching face with the DB stored image'}
                        logging.info("====result", result)
                        # Check for face match with images in the image_database
                        for known_name, known_encoding in image_database.items():
                            distances = face_recognition.face_distance([known_encoding], face_encodings[0])

                            # Choose a suitable threshold for confidence
                            confidence_threshold = 0.6
                            distance_threshold = 0.5  # Set your desired face distance threshold here

                            # Check if the distance is below the threshold
                            if distances[0] < distance_threshold:
                                result = {'matched': True, 'name': known_name, 'confidence': 1 - distances[0],
                                          'message': 'Match Found!'}
                                logging.info("====result", result)
                                image_database.clear()
                                logging.info("=====imagedatabase %s:", len(image_database))
                                print("===result", result)
                                break

                        image_database.clear()
                        logging.info("=====imagedatabase %s:", len(image_database))
                        socketio.emit('face_recognition_result', result)

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








# ... (rest of the code remains unchanged)


#
# #Event handler for streaming frames from the client
# @socketio.on('stream_frame')
# def handle_webcam_frame(data):
#     try:
#         # Decode the base64 encoded image
#         frame_data = data
#         binary_data = base64.b64decode(frame_data)
#         frame_np = None  # Set a default value before the 'try' block
#         frameArray = np.frombuffer(binary_data, dtype=np.uint8)
#
#         try:
#             frame_np = cv2.imdecode(frameArray, cv2.IMREAD_COLOR)
#         except Exception as e:
#             print(f"Error decoding frame: {e}")
#
#         # Check if 'frame_np' is not None before using it
#         if frame_np is not None:
#             # Perform anti-spoofing test using the 'test' function
#             label, confidence = test(image=frame_np, model_dir=os.path.join("resources", "anti_spoof_models"), device_id=0)
#
#             spoofing_threshold = 0.5
#
#             if label == 1 and confidence > spoofing_threshold:
#                 # Proceed with face recognition if not spoofed
#                 rgb_frame = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
#
#                 # Detect faces in the frame
#                 face_locations = face_recognition.face_locations(rgb_frame)
#                 face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
#                 print("===>face_encodings_live", face_encodings)
#                 # Initialize result dictionary
#                 result = {'matched': False, 'message': 'not matching face with the DB stored image'}
#
#                 for face_encoding in face_encodings:
#                     # Check for face match with images in the image_database
#                     matches = face_recognition.compare_faces(list(image_database.values()), face_encoding)
#                     print("===>matches", matches)
#                     logging.info("matches: %s", matches)
#                     # If a match is found, update the result dictionary
#                     if True in matches:
#                         first_match_index = matches.index(True)
#                         name = list(image_database.keys())[first_match_index]
#                         print("===name", name)
#                         result = {'matched': True,  'message': 'Match Found!'}
#                         logging.info("Name: %s", result)
#                         break
#
#                 # Emit the results to the connected client
#                 socketio.emit('face_recognition_result', result)
#             else:
#                 socketio.emit('face_recognition_result', {'matched': False,'message': 'please provide real face'})
#                 logging.error("please provide real face")
#         else:
#             # Emit an appropriate response to the client
#             socketio.emit('face_recognition_result', {'matched': False,'message': 'failed to detect the face'})
#             logging.error("frame does not detect a proper face")
#
#     except Exception as e:
#         # Emit an error response to the client
#         socketio.emit('error', {'message': str(e)})
#         logging.error(f"Error: {e}")


# Main entry point


if __name__ == '__main__':
    socketio.run(app, host="0.0.0.0", debug=True, port=5001)
    video_capture.release()
    cv2.destroyAllWindows()
    image_database.clear()

# # Event handler for streaming frames from the client
# @socketio.on('stream_frame')
# def handle_webcam_frame(data):
#     try:
#         # Decode the base64 encoded image
#         # frame_data = data
#         print("===frame", data[:9])
#         binary_data = base64.b64decode(data)
#         frame_np = cv2.imdecode(np.frombuffer(binary_data, dtype=np.uint8), cv2.IMREAD_COLOR)
#
#         # Check if 'frame_np' is not None before using it
#         if frame_np is not None:
#             # Perform anti-spoofing test using the 'test' function
#             label, confidence = test(image=frame_np, model_dir=os.path.join("resources", "anti_spoof_models"), device_id=0)
#
#             spoofing_threshold = 0.5
#
#             if label == 1 and confidence > spoofing_threshold:
#                 # Proceed with face recognition if not spoofed
#                 rgb_frame = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
#
#                 # Detect faces in the frame
#                 face_locations = face_recognition.face_locations(rgb_frame)
#
#                 if len(face_locations) > 0:
#                     # Extract the first face location
#
#                     # Extract the face encoding for the first detected face
#                     face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
#                     print("===>into frame face_encoding", len(face_encoding))
#                     # Initialize result dictionary
#                     result = {'matched': False, 'message': 'not matching face with the DB stored image'}
#
#                     # Check for face match with images in the image_database
#                     matches = face_recognition.compare_faces(list(image_database.values()), face_encoding)
#                     print("==matches", matches)
#                     logging.info("matches: %s", matches)
#
#                     # If a match is found, update the result dictionary
#                     if True in matches:
#                         result = {'matched': True, 'message': 'Match Found!'}
#                         logging.info("Name: %s", result)
#                     else:
#                         # Emit an appropriate response to the client if no face is matched
#                         result = {'matched': False, 'message': 'not matching face with the DB stored image'}
#                         logging.info("Real face detected, but no match found")
#                 else:
#                     # Emit an appropriate response to the client if no face is detected
#                     result = {'matched': False, 'message': 'No face detected'}
#                     logging.info("No face detected")
#
#                 # Emit the results to the connected client
#                 socketio.emit('face_recognition_result', result)
#             else:
#                 socketio.emit('face_recognition_result', {'matched': False, 'message': 'please provide real face'})
#                 logging.error("please provide a real face")
#         else:
#             # Emit an appropriate response to the client if 'frame_np' is None
#             socketio.emit('face_recognition_result', {'matched': False, 'message': 'failed to decode the frame'})
#             logging.error("failed to decode the frame")
#
#     except Exception as e:
#         # Emit an error response to the client
#         socketio.emit('error', {'message': str(e)})
#         logging.error(f"Error: {e}")

# ... (rest of the code remains unchanged)
