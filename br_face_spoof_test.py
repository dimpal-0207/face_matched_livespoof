#final project app for realtime face detection and antispoofing frontend into index.html

import base64
import datetime
import logging
import boto3
import cv2
import face_recognition
import numpy as np
from flask import Flask, render_template
from flask_cors import CORS
from flask_socketio import SocketIO
from decouple import config
import time

# Import the 'test' function from your existing code
from test import test

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize Flask app and SocketIO
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, ping_timeout=60000, ping_interval=20000)

# AWS S3 credentials
AWS_ACCESS_KEY = config("AWS_ACCESS_KEY")
AWS_SECRET_KEY = config("AWS_SECRET_KEY")
BUCKET_NAME = config("BUCKET_NAME")
FOLDER_NAME = config("FOLDER_NAME")

# Initialize AWS S3 client
s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)

# List objects in the specified folder
response_ = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=FOLDER_NAME)
image_database = {}

# Iterate over objects in the folder
for obj in response_.get('Contents', []):
    object_key = obj['Key']
    if object_key.lower().endswith(('.png', '.jpg', '.jpeg')):
        # print(f"Processing Image File: {object_key}")

        # Download the image
        local_image_path = f"{object_key}"
        s3.download_file(BUCKET_NAME, object_key, local_image_path)

        # Load image and extract face encoding
        image = face_recognition.load_image_file(local_image_path)
        face_encoding = face_recognition.face_encodings(image)[0]  # Assuming there is only one face in each image

        # Extract the name from the object key (customize based on your naming conventions)
        face_name = object_key.split('/')[-1].split('.')[0]

        # Add face encoding to the image_database
        image_database[face_name] = face_encoding

# OpenCV setup for live face detection
video_capture = cv2.VideoCapture(0)

# Route for rendering the index.html file


@app.route('/')
def index():
    return render_template('index.html')

# Event handler for client connection


@socketio.on('connect')
def handle_connect():
    logging.info("Client connected")
    socketio.emit('connect', {'status': 'connected with server'})

# Event handler for client disconnection


@socketio.on('disconnect')
def handle_disconnect():
    logging.info("Client disconnected")
    socketio.emit('disconnect_response', {'status': 'disconnected'})

# Event handler for streaming frames from the client
import dlib


# Function to extract facial features (landmarks and beard region)
def extract_face_features(image):
    # Load the pre-trained facial landmark predictor from dlib
    predictor_path = "model_dlib/shape_predictor_68_face_landmarks.dat"

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = detector(gray)

    # Extract facial landmarks for the first face (assuming one face per image)
    if faces:
        landmarks = predictor(gray, faces[0])

        # Extract coordinates of specific facial landmarks (customize as needed)
        beard_region = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(28, 31)]

        return {'landmarks': landmarks, 'beard_region': beard_region}
    else:
        return None


# Update the handle_webcam_frame function
@socketio.on('stream_frame')
def handle_webcam_frame(data):
    try:
        frame_data = data.split(',')[1]
        binary_data = base64.b64decode(frame_data)
        frame_np = None

        try:
            frame_np = cv2.imdecode(np.frombuffer(binary_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"Error decoding frame: {e}")

        if frame_np is not None:
            # Perform anti-spoofing test using the 'test' function
            label, confidence = test(image=frame_np,
                                     model_dir=r"resources\anti_spoof_models",
                                     device_id=0)
            logging.info("===>label: %s", label)
            logging.info("===>confidence : %s", confidence)

            spoofing_threshold = 0.5

            if label == 1 and confidence > spoofing_threshold:
                # Extract facial features
                features = extract_face_features(frame_np)

                if features:
                    landmarks = features['landmarks']
                    # print("====landmark", landmarks)
                    beard_region = features['beard_region']
                    # print("===>beardregion", beard_region)
                    # Combine face encodings with additional features
                    rgb_frame = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(rgb_frame)
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                    # Initialize result dictionary
                    result = {'matched': False, 'name': 'Unknown', 'spoofing': 'Spoofed'}

                    for face_encoding in face_encodings:
                        # Check for face match with images in the image_database
                        matches = face_recognition.compare_faces(list(image_database.values()), face_encoding,
                                                                 tolerance=0.5)

                        if True in matches:
                            first_match_index = matches.index(True)
                            name = list(image_database.keys())[first_match_index]
                            logging.info("Name: %s", name)
                            result = {'matched': True, 'name': name, 'spoofing': 'Real face'}
                            break

                    # Emit the results to the connected clients
                    socketio.emit('face_recognition_result', result)
                else:
                    socketio.emit('face_recognition_result',
                                  {'matched': False, 'name': 'Unknown', 'spoofing': 'Spoofed face'})
                    logging.error("Frame does not detect a proper face")
            else:
                socketio.emit('face_recognition_result',
                              {'matched': False, 'name': 'Unknown', 'spoofing': 'Spoofed face'})
                logging.error("Frame does not detect a proper face")

    except Exception as e:
        logging.error(f"Error: {e}")


if __name__ == '__main__':
    socketio.run(app, debug=True, port=8080)
    video_capture.release()
    cv2.destroyAllWindows()
