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
from flask import Flask, render_template, request, session
from flask_cors import CORS
from flask_socketio import SocketIO
import atexit
import dlib

import asyncio
import time
# Import the 'test' function from your existing code
from test import test
indian_timezone = timezone('Asia/Kolkata')

# Configure logging
logging.basicConfig(level=logging.INFO, filename="file.log", filemode="a", format="%(asctime)s%(levelname)s:%(name)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S %Z" ) # Customize the date format he)
logger = logging.getLogger()


app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
CORS(app)
sio = SocketIO(app, supports_credentials=True, ping_interval=10000, ping_timeout=5000, reconnection=True, cors_allowed_origins="*", cookie=False)
# sio = socketio.Server()
user_data = {}


@app.route('/')
def index():
    return render_template("index.html")


image_database = {}


@sio.on('connect')
def handle_connect():
    sid = request.sid
    logging.info("Client connected")
    sio.emit('connect', {'status': 'connected with server'},room=sid)


@sio.on('get_face')
def face_data(data):
    # print("====data", len(data))
    sid = request.sid
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
                        sio.emit('get_face', {"status": 200, 'message': 'Face Detected!'})
                else:
                    logging.info('encode len  of facedata: %s', "Not get face encodings")
                    sio.emit('get_face', {"status": 400, 'message': 'Please upload a proper face image'}, sid=sid)
        else:
            sio.emit('get_face', {"status": 400,'message': 'Not received face data'}, sid=sid)
    except Exception as e:
        logging.error(f"Error: {e}")


@sio.on('join_Room')
def joinroom(data):
    image_databases = {}
    logging.info("userid_data: %s", data)
    print("===data", data)
    sid = request.sid
    print("====sid into join room event", sid)
    user_id = data
    if user_id not in image_database:
        image_database[user_id] = {}

    user_data[sid] = user_id
    print("===user_id", data)
    try:
        # Hit the API with the userid
        api_url = f"http://13.126.129.218:6002/api/auth/user-profile-image/{data}"
        response = requests.get(api_url)
        # print("===response", response)
        logging.info('response: %s', response)
        # Check the API response and perform actions accordingly
        if response.status_code == 200:
            api_data = response.json()
            profile_image_data = api_data.get('data', [])[0].get('profileImage', None)
            face_name = api_data.get('data', [])[0].get('profileImagePath', None).split('/')[-1].split('.')[0]
            binary_data = base64.b64decode(profile_image_data)
            image_np = cv2.imdecode(np.frombuffer(binary_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            face_encoding = face_recognition.face_encodings(image_np)[0]
            logging.info('faceencoding of api face: %s', len(face_encoding))
            image_database[user_id][sid] = face_encoding
            print("===image_databases", image_database)
            # print("===image_database", image_database)
            return image_database
    except Exception as e:
        logging.error(f"Error: {e}")


face_recognition_model = dlib.get_frontal_face_detector()



# Specify the desired width and height
width = 640  # Set your desired width
height = 480  # Set your desired height


# working=====>
@sio.on('stream_frame')
def handle_webcam_frame(data):
    # print("=====frame", data)
    sid = request.sid
    print("===sid in stream frame event", sid)
    try:
        # user_id = request.sid
        if 'anti_spoofing_in_progress' in session:
            return
        # user_data_frame[""] = data
        # print("===user_data_frame", user_data_frame)
        session['anti_spoofing_in_progress'] = True
        # Decode the base64 encoded image
        # data = data.split(',')[1]
        # frame = await asyncio.create_task(data["data"])
        # print("====<frame", frame)
        binary_data = base64.b64decode(data["data"])
        frame_np = cv2.imdecode(np.frombuffer(binary_data, dtype=np.uint8), cv2.IMREAD_COLOR)

        # Check if 'frame_np' is not None before using it
        if frame_np is not None:
            # Perform anti-spoofing test using the 'test' function
            label, confidence = test(image=frame_np, model_dir=os.path.join("resources", "anti_spoof_models"), device_id=0)

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

                        # Check for face match with images in the image_database
                        for user_id , known_encoding in image_database.items():
                            print("==>imagedatabase in for loop", image_database)
                            distances = face_recognition.face_distance([known_encoding], face_encodings[0])

                            # Choose a suitable threshold for confidence
                            confidence_threshold = 0.6
                            distance_threshold = 0.7  # Set your desired face distance threshold here

                            # Check if the distance is below the threshold
                            if distances[0] < distance_threshold:
                                result = {'matched': True, 'name': user_id, 'confidence': 1 - distances[0],
                                          'message': 'Match Found!'}
                                break

                        sio.emit('face_recognition_result', result, room=sid)
                        print("====sioemit result", sid, result)
                        logging.info("====result %s", result)

                        print("result:", result)
                else:
                    result = {'matched': False, 'message': 'No face detected!!'}
                    sio.emit('face_recognition_result', result , room=sid)
                    logging.info("====result %s:", result)
            else:
                result = {'matched': False, 'message': 'Please provide a real face!!'}
                logging.info("====result %s", result)
                sio.emit('face_recognition_result', result, room=sid)
        else:
            result = {'matched': False, 'message': 'Failed to decode the frame'}
            logging.info("====result %s", result)
            sio.emit('face_recognition_result', result, room=sid)
        # task = socketio.start_background_task(handle_webcam_frame)
        # await task.wait()
    except Exception as e:
        sio.emit('error', {'message': str(e)})
        logging.error(f"Error: {e}")
    finally:
        session.pop('anti_spoofing_in_progress', None)

# Event handler for user disconnecting


@sio.on('disconnect')
def handle_disconnect():
    logging.info("Client disconnected")
    sio.emit('disconnect_response', {'status': 'disconnected'})
    session.pop('anti_spoofing_in_progress', None)


def cleanup():
    image_database.clear()


if __name__ == '__main__':
    atexit.register(cleanup)
    sio.run(app, host="0.0.0.0", debug=True, port=5002)



