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
    if object_key.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
        print(f"Processing Image File: {object_key}")

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
@socketio.on('stream_frame')
def handle_webcam_frame(data):
    try:
        # Decode the base64 encoded image
        frame_data = data.split(',')[1]
        binary_data = base64.b64decode(frame_data)
        frame_np = None  # Set a default value before the 'try' block
        frameArray = np.frombuffer(binary_data, dtype=np.uint8)

        try:
            frame_np = cv2.imdecode(frameArray, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"Error decoding frame: {e}")

        # Check if 'frame_np' is not None before using it
        if frame_np is not None:
            # Perform anti-spoofing test using the 'test' function
            label = test(image=frame_np, model_dir='/path/to/antispoofing/model', device_id=0)

            if label == 1:
                # Proceed with face recognition if not spoofed
                rgb_frame = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)

                # Detect faces in the frame
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                for face_encoding in face_encodings:
                    # Check for face match with images in the image_database
                    matches = face_recognition.compare_faces(list(image_database.values()), face_encoding)

                    name = "Unknown"  # Default name if no match found

                    # If a match is found, use the name from the image_database
                    if True in matches:
                        first_match_index = matches.index(True)
                        name = list(image_database.keys())[first_match_index]
                        logging.info("Name: %s", name)
                        # Emit the results to the connected clients
                        socketio.emit('face_recognition_result', {'matched': True, 'name': name})
                    else:
                        logging.info("Frame is empty")
                        # Send an appropriate response to the client
                        socketio.emit('face_recognition_result', {'matched': False, 'name': 'Unknown'})
            else:
                # Spoofing detected, emit a response
                logging.warning("Spoofing detected!")
                socketio.emit('face_recognition_result', {'matched': False, 'name': 'Spoofed'})

        else:
            socketio.emit('face_recognition_result', {'matched': False, 'name': 'Unknown'})
            logging.error("frame does not detect a proper face")
    except Exception as e:
        logging.error(f"Error: {e}")

# Main entry point
if __name__ == '__main__':
    socketio.run(app, debug=True, port=8080)
    video_capture.release()
    cv2.destroyAllWindows()
