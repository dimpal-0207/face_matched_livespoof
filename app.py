import base64
import logging
import os

import boto3
import botocore
import cv2
import face_recognition
import numpy as np
import requests
from flask import Flask, render_template, request, session
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room
from decouple import config
from datetime import datetime, timezone, timedelta

# Import the 'test' function from your existing code
from numpy import size
from cachetools import TTLCache
from test import test
india_time_zone = timezone(timedelta(hours=5, minutes=30))

# Configure logging
logging.basicConfig(level=logging.INFO, filename="logs.log", filemode="a", format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger()

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, ping_interval=10000, ping_timeout=5000, reconnection=True, cors_allowed_origins="*", cookie=False)

# AWS S3 credentials
AWS_ACCESS_KEY = config("AWS_ACCESS_KEY")
AWS_SECRET_KEY = config("AWS_SECRET_KEY")
BUCKET_NAME = config("BUCKET_NAME")
FOLDER_NAME = config("FOLDER_NAME")
TARGET_FILENAME = "dimpal.png"


def process_s3_image(aws_access_key, aws_secret_key, bucket_name, folder_name, target_filename):
    try:
        # Initialize AWS S3 client
        s3 = boto3.client('s3', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key)

        # Check if the object (file) exists in the S3 bucket
        s3.head_object(Bucket=bucket_name, Key=os.path.join(folder_name, target_filename))

        # If the object exists, proceed with processing
        # print(f"Processing Image File: {target_filename}")

        # Create a directory for images if it doesn't exist
        image_folder = "images"
        os.makedirs(image_folder, exist_ok=True)

        # Download the image
        local_image_path = os.path.join(image_folder, target_filename)
        print("===local", local_image_path)
        s3.download_file(bucket_name, os.path.join(folder_name, target_filename), local_image_path)

        # Load image and extract face encoding (replace this with your actual processing logic)
        image = face_recognition.load_image_file(local_image_path)
        # print("===imagetype", image)
        face_encoding = face_recognition.face_encodings(image)[0]  # Assuming there is only one face in the image
        # print("===face_encoding", face_encoding)
        # print("===face_encodingtype", type(face_encoding))
        # Extract the name from the object key (customize based on your naming conventions)
        face_name = target_filename.split('.')[0]
        # print("===face_name", face_name)
        # Add face encoding to the image_database
        image_database = {face_name: face_encoding}

        # print("===image_database", type(image_database))
        return image_database

    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            print(f"Error: The object {os.path.join(folder_name, target_filename)} does not exist in the S3 bucket.")
        else:
            print(f"Error: {e}")
        return {}

anti_spoof_cache = TTLCache(maxsize=100, ttl=60)  # Cache anti-spoofing test results for 60 seconds
# Example usage
image_database = process_s3_image(AWS_ACCESS_KEY, AWS_SECRET_KEY, BUCKET_NAME, FOLDER_NAME, TARGET_FILENAME)
# print("Image Database:", list(image_database.values()))

# OpenCV setup for live face detection
video_capture = cv2.VideoCapture(0)

# Route for rendering the index.html file
@app.route('/')
def index():
    return render_template("index.html")

active_sessions ={}
# Event handler for client connection
@socketio.on('connect')
def handle_connect(data):
    print("===data", data)
    user_id = data
    session["user_id"] = user_id
    if user_id in session:
        active_sessions[user_id] = request.sid
        join_room(user_id)
        socketio.emit('message', {'data': f'Connected as {user_id}'})
    logging.info(f"User {user_id} connected")


@socketio.on('message')
def handle_message(msg):
    print('Received message:', msg)
    msg = "abcd"
    socketio.emit('message', msg)


# Event handler for client disconnection
@socketio.on('disconnect')
def handle_disconnect():
    user_id = request.sid
    logging.info(f"User {user_id} disconnected")

match_found = False

@socketio.on('stream_frame')
def handle_webcam_frame(data):
    global match_found  # Declare the flag as a global variable
    try:
        frame_data = data.split(',')[1]
        binary_data = base64.b64decode(frame_data)
        frame_np = cv2.imdecode(np.frombuffer(binary_data, dtype=np.uint8), cv2.IMREAD_COLOR)

        if frame_np is not None:
            label, confidence = test(image=frame_np, model_dir=os.path.join("resources", "anti_spoof_models"), device_id=0)
            spoofing_threshold = 0.5

            if label == 1 and confidence > spoofing_threshold:
                rgb_frame = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)

                if len(face_locations) > 0:
                    face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
                    result = {'matched': False, 'message': 'not matching face with the DB stored image'}

                    matches = face_recognition.compare_faces(list(image_database.values()), face_encoding)

                    if True in matches and not match_found:
                        match_found = True  # Set the flag to True
                        name = list(image_database.keys())[matches.index(True)]
                        result = {'matched': True, 'name': name, 'message': 'Match Found!'}
                        logging.info("Name: %s", result)

                        # Emit the results to the connected client
                        socketio.emit('face_recognition_result', result)

                else:
                    result = {'matched': False, 'message': 'not matching face with the DB stored image'}
                    logging.info("No face detected")

                # Emit the results to the connected client only if a match has not been found
                if not match_found:
                    socketio.emit('face_recognition_result', result)
            else:
                socketio.emit('face_recognition_result', {'matched': False, 'message': 'please provide real face'})
                logging.error("please provide a real face")
        else:
            socketio.emit('face_recognition_result', {'matched': False, 'message': 'failed to decode the frame'})
            logging.error("failed to decode the frame")

    except Exception as e:
        socketio.emit('error', {'message': str(e)})
        logging.error(f"Error: {e}")

# @socketio.on('stream_frame')
# def handle_webcam_frame(data):
#     try:
#         frame_data = data.split(',')[1]
#         # Decode the base64 encoded image
#         binary_data = base64.b64decode(frame_data)
#         frame_np = cv2.imdecode(np.frombuffer(binary_data, dtype=np.uint8), cv2.IMREAD_COLOR)
#
#         # Check if 'frame_np' is not None before using it
#         if frame_np is not None:
#             # Perform anti-spoofing test using the 'test' function
#             label, confidence = test(image=frame_np, model_dir=os.path.join("resources", "anti_spoof_models"),
#                                      device_id=0)
#
#             spoofing_threshold = 0.5
#
#             if label == 1 and confidence > spoofing_threshold:
#                 # Proceed with face recognition if not spoofed
#                 rgb_frame = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
#
#                 # Detect faces in the frame
#                 face_locations = face_recognition.face_locations(rgb_frame)
#                 face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)[0]
#                 matches = face_recognition.compare_faces(list(image_database.values()), face_encodings)
#                 print("===matches", matches)
#                 logging.info("matches: %s", matches)
#                 # name = "" if matches[0] else "Unknown"
#                 # If a match is found, update the result dictionary
#                 if matches:
#                     # name = list(image_database.keys())
#                     # print("===name", name)
#                     # logging.info("Name: %s", name)
#                     result = {'matched': True,'message':'Match Found!'}
#                     socketio.emit('face_recognition_result', result)
#                     logging.info("Name: %s", result)
#
#                 else:
#                     socketio.emit('face_recognition_result',
#                                   {'matched': False, 'message': 'not matching face with the DB stored image'})
#                 # Emit the results to the connected client
#
#             else:
#                 # Emit an appropriate response to the client if no face is detected
#                 socketio.emit('face_recognition_result',
#                               {'matched': False, 'message': 'not matching face with the DB stored image'})
#         else:
#             socketio.emit('face_recognition_result', {'matched': False, 'message': 'Please provide a real face'})
#             logging.error("Please provide a real face")
#
#     except Exception as e:
#         # Emit an error response to the client
#         socketio.emit('error', {'message': str(e)})
#         logging.error(f"Error: {e}")



# @socketio.on('stream_frame')
# def handle_webcam_frame(data):
#     try:
#         # Validate if the data is in the expected format
#         # if not isinstance(data, str) or not data.startswith('data:image'):
#         #     raise ValueError('Invalid data format')
#
#         # Decode the base64 encoded image
#         frame_data = data.split(',')[1]
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
#             face_recognition_threshold=0.8
#
#             if label == 1 and confidence > spoofing_threshold:
#                 # Proceed with face recognition if not spoofed
#                 rgb_frame = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
#
#                 # Detect faces in the frame
#                 face_locations = face_recognition.face_locations(rgb_frame)
#                 face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
#                 # print("===face_encoding", face_encodings)
#                 # print("===lenoffaceencoding", len(face_encodings))
#                 if len(face_encodings) == 1:
#                     # Initialize result dictionary
#                     # result = {'matched': False, 'name': 'Unknown', 'message': 'not matching face with the DB stored image'}
#
#                     for face_encoding in face_encodings:
#                         print("===end into loop", face_encoding[0])
#                         # Check for face match with images in the image_database
#                         matches = face_recognition.compare_faces(list(image_database.values()), face_encoding)
#
#                         # If a match is found, update the result dictionary
#                         if True in matches:
#                             first_match_index = matches.index(True)
#                             name = list(image_database.keys())[first_match_index]
#                             logging.info("Name: %s", name)
#                             result = {'matched': True, 'name': name, 'message': 'Match Found!'}
#                             emit('face_recognition_result', result)
#
#                             break
#
#                         else:
#                             result = {'matched': False, 'name': 'Unknown',
#                                       'message': 'not matching face with the DB stored image'}
#                             emit("face_recognition_result", result)
#                     # Emit the results to the connected client
#
#             else:
#                 emit('face_recognition_result', {'matched': False, 'name': 'Unknown', 'message': 'please provide real face'})
#                 logging.error("frame does not detect a proper face")
#         else:
#             # Emit an appropriate response to the client
#             emit('face_recognition_result', {'matched': False, 'name': 'Unknown','message': 'failed to detect the face'})
#             logging.error("frame does not detect a proper face")
#
#     except Exception as e:
#         # Emit an error response to the client
#         emit('error', {'message': str(e)})
#         logging.error(f"Error: {e}")


# Main entry point
if __name__ == '__main__':
    socketio.run(app, host="0.0.0.0", debug=True, port=5000)
    video_capture.release()
    cv2.destroyAllWindows()

# @socketio.on('stream_frame')
# def handle_webcam_frame(data):
#     global processing_face
#
#     try:
#         if processing_face:
#             return  # Ignore new frames if a face is already being processed
#
#         processing_face = True  # Set the flag to indicate face processing is in progress
#
#         frame_data = data.split(',')[1]
#         binary_data = base64.b64decode(frame_data)
#         frame_np = cv2.imdecode(np.frombuffer(binary_data, dtype=np.uint8), cv2.IMREAD_COLOR)
#
#         script_directory = os.path.dirname(os.path.abspath(__file__))
#         resources_directory = os.path.join(script_directory, "resources", "anti_spoof_models")
#
#         if frame_np is not None:
#             label, confidence = test(image=frame_np, model_dir=resources_directory, device_id=0)
#             spoofing_threshold = 0.5
#
#             if label == 1 and confidence > spoofing_threshold:
#                 rgb_frame = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
#                 face_locations = face_recognition.face_locations(rgb_frame)
#                 face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
#
#                 result = {'matched': False, 'name': 'Unknown', 'message': 'not matching face with the DB stored image'}
#
#                 if len(face_encodings) == 1:  # Ensure only one face is detected
#                     for face_encoding in face_encodings:
#                         matches = face_recognition.compare_faces(list(image_database.values()), face_encoding)
#
#                         if True in matches:
#                             first_match_index = matches.index(True)
#                             name = list(image_database.keys())[first_match_index]
#                             logging.info("Name: %s", name)
#                             result = {'matched': True, 'name': name, 'message': 'Match Found!'}
#                             break
#
#                 socketio.emit('face_recognition_result', result)
#             else:
#                 socketio.emit('face_recognition_result', {'matched': False, 'name': 'Unknown', 'message': 'please provide real face'})
#                 logging.error("frame does not detect a proper face")
#         else:
#             socketio.emit('face_recognition_result', {'matched': False, 'name': 'Unknown', 'message': 'failed to detect the face'})
#             logging.error("frame does not detect a proper face")
#
#     except Exception as e:
#         socketio.emit('error', {'message': str(e)})
#         logging.error(f"Error: {e}")
#     finally:
#         processing_face = False  #


# Main entry point+
if __name__ == '__main__':
    socketio.run(app, host="0.0.0.0", debug=True, port=5001)
    video_capture.release()
    cv2.destroyAllWindows()



















# @socketio.on('stream_frame')
# def handle_webcam_frame(data):
#     print("======================================>")
#     print("===data", data)
#     print("========================================>")
#     try:
#         # Decode the base64 encoded image
#         frame_data = data.split(',')[1]
#         # print("===>frame", frame_data)
#         binary_data = base64.b64decode(frame_data)
#         frame_np = None  # Set a default value before the 'try' block
#         frameArray = np.frombuffer(binary_data, dtype=np.uint8)
#
#         try:
#             frame_np = cv2.imdecode(frameArray, cv2.IMREAD_COLOR)
#         except Exception as e:
#             print(f"Error decoding frame: {e}")
#
#         # Get the directory of the current script
#
#         script_directory = os.path.dirname(os.path.abspath(__file__))
#         print("scriptpath", script_directory)
#         # Construct the dynamic path relative to the script's location
#
#         resources_directory = os.path.join(script_directory, "resources", "anti_spoof_models")
#         print("===resou", resources_directory)
#         # Check if 'frame_np' is not None before using it
#         if frame_np is not None:
#             # Perform anti-spoofing test using the 'test' function
#             label, confidence = test(image=frame_np,
#                                 model_dir=resources_directory,
#                                 device_id=0)
#             # logging.info("===>lable: %s", label)
#             # logging.info("===>confidence : %s", confidence)
#             spoofing_threshold = 0.5
#
#             if label == 1 and confidence > spoofing_threshold:
#                 # Proceed with face recognition if not spoofed
#                 rgb_frame = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
#
#                 # Detect faces in the frame
#                 face_locations = face_recognition.face_locations(rgb_frame)
#                 face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
#
#                 # Initialize result dictionary
#                 result = {'matched': False, 'name': 'Unknown', 'message': 'please provide real face '}
#
#                 for face_encoding in face_encodings:
#                     # Check for face match with images in the image_database
#                     matches = face_recognition.compare_faces(list(image_database.values()), face_encoding)
#
#                     # If a match is found, update the result dictionary
#                     if True in matches:
#                         first_match_index = matches.index(True)
#                         name = list(image_database.keys())[first_match_index]
#                         logging.info("Name: %s", name)
#                         result = {'matched': True, 'name': name, 'message': 'Match Found!'}
#                         break
#
#                 # Emit the results to the connected clients
#                 socketio.emit('face_recognition_result', result)
#             else:
#                 socketio.emit('face_recognition_result', {'matched': False, 'name': 'Unknown', 'message': 'not matching face with the DB stored image'})
#                 logging.error("frame does not detect a proper face")
#         else:
#             # Emit an appropriate response to the client
#             socketio.emit('face_recognition_result', {'matched': False, 'name': 'Unknown','message': 'failed to detect the face'})
#             logging.error("frame does not detect a proper face")
#
#     except Exception as e:
#         logging.error(f"Error: {e}")


