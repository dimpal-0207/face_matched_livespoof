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


# Import the 'test' function from your existing code
from test import test
indian_timezone = timezone('Asia/Kolkata')

# Configure logging
logging.basicConfig(level=logging.INFO, filename="file.log", filemode="a", format="%(asctime)s%(levelname)s:%(name)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S %Z" ) # Customize the date format he)
logger = logging.getLogger()


app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
CORS(app)
socketio = SocketIO(app, ping_interval=10000, ping_timeout=5000, reconnection=True, cors_allowed_origins="*", cookie=False)

user_data = {}



@app.route('/')
def index():
    return render_template("index.html")


image_database = {}

def write_user_data_to_csv():
    data_dir = "user_data_csv"  # Change this to your desired directory
    os.makedirs(data_dir, exist_ok=True)  # Create the directory if it doesn't exist

    for user_id, user_df in user_data.items():
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        csv_filename = os.path.join(data_dir, f"{user_id}_data_{timestamp}.csv")
        user_df.to_csv(csv_filename, index=False)


@socketio.on('connect')
def handle_connect():
    user_id = request.sid

    # if user_id not in user_data:
    #     # Create a new dataframe for the user
    #     user_data[user_id] = pd.DataFrame(columns=['user_id','Column1', 'Column2'])
    print(f"====client Connected with userid {user_id}")
    logging.info("Client connected")
    socketio.emit('connect', {'status': 'connected with server'})

import dlib

@socketio.on('get_face')
def face_data(data):
    # print("====data", len(data))
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
    print("===data", data)
    user_id = data
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
            profile_image_data = api_data.get('data', [])[0].get('profileImage', None)
            face_name = api_data.get('data', [])[0].get('profileImagePath', None).split('/')[-1].split('.')[0]
            binary_data = base64.b64decode(profile_image_data)
            image_np = cv2.imdecode(np.frombuffer(binary_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            face_encoding = face_recognition.face_encodings(image_np)[0]
            logging.info('faceencoding of api face: %s', len(face_encoding))
            image_database = {user_id : face_encoding}
            return image_database
    except Exception as e:
        logging.error(f"Error: {e}")


face_recognition_model = dlib.get_frontal_face_detector()

import asyncio

import time

# Specify the desired width and height
width = 640  # Set your desired width
height = 480  # Set your desired height
user_data_frame = {}
import pandas as pd
# working=====>
@socketio.on('stream_frame')
def handle_webcam_frame(data):
    # print("=====frame", data)
    try:
        # user_id = request.sid
        if 'anti_spoofing_in_progress' in session:
            return
        user_data_frame[""] = data
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
                            distances = face_recognition.face_distance([known_encoding], face_encodings[0])

                            # Choose a suitable threshold for confidence
                            confidence_threshold = 0.6
                            distance_threshold = 0.7  # Set your desired face distance threshold here

                            # Check if the distance is below the threshold
                            if distances[0] < distance_threshold:
                                result = {'matched': True, 'name': user_id, 'confidence': 1 - distances[0],
                                          'message': 'Match Found!'}
                                break

                        # Update user's dataframe with new result
                        # user_df = user_data.get(user_id, pd.DataFrame(columns=['Column1', 'Column2']))
                        # print("===userdf", user_df)
                        # new_row = pd.Series({'user_id': user_id, 'Column1': result, 'Column2': result.get('confidence', 0.0)})
                        # user_df = pd.concat([user_df, new_row.to_frame().T], ignore_index=True)
                        # user_data[user_id] = user_df

                        socketio.emit('face_recognition_result', result)
                        logging.info("====result %s", result)
                        print("result:", result)

            else:
                result = {'matched': False, 'message': 'Please provide a real face'}
                logging.info("====result %s", result)
                socketio.emit('face_recognition_result', result)
        else:
            result = {'matched': False, 'message': 'Failed to decode the frame'}
            logging.info("====result %s", result)
            socketio.emit('face_recognition_result', result)
        # task = socketio.start_background_task(handle_webcam_frame)
        # await task.wait()
    except Exception as e:
        socketio.emit('error', {'message': str(e)})
        logging.error(f"Error: {e}")
    finally:
        session.pop('anti_spoofing_in_progress', None)

# Event handler for user disconnecting
print("===user_daata", user_data_frame)

@socketio.on('disconnect')
def handle_disconnect():
    logging.info("Client disconnected")
    socketio.emit('disconnect_response', {'status': 'disconnected'})
    session.pop('anti_spoofing_in_progress', None)


def cleanup():
    image_database.clear()

# Periodically write user data to CSV (adjust the interval as needed)
# from apscheduler.schedulers.background import BackgroundScheduler
#
# scheduler = BackgroundScheduler()
# scheduler.add_job(write_user_data_to_csv, 'interval', seconds=10)  # Adjust the interval as needed
# scheduler.start()


if __name__ == '__main__':
    atexit.register(cleanup)
    socketio.run(app, host="0.0.0.0", debug=True, port=5002)



# # Event handler for streaming frames from the client
# @socketio.on('stream_frame')
# def handle_webcam_frame(data):
#     try:
#         # Check if an anti-spoofing test is already in progress
#         if 'anti_spoofing_in_progress' in socketio.session:
#             return
#
#         # Mark that an anti-spoofing test is in progress
#         socketio.session['anti_spoofing_in_progress'] = True
#
#         # Decode the base64 encoded image
#         binary_data = base64.b64decode(data)
#         frame_np = cv2.imdecode(np.frombuffer(binary_data, dtype=np.uint8), cv2.IMREAD_COLOR)
#
#         # Perform anti-spoofing test using the 'test' function
#         label, confidence = test(image=frame_np, model_dir=os.path.join("resources", "anti_spoof_models"),
#                                  device_id=0)
#
#         # Check if the frame is not None and anti-spoofing test passed
#         if frame_np is not None and label == 1:
#             # Proceed with face recognition
#             rgb_frame = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
#             face_locations = face_recognition_model(frame_np, 1)
#
#             if len(face_locations) > 0:
#                 # Extract face encodings for all detected faces
#                 for face_location in face_locations:
#                     top, right, bottom, left = face_location.top(), face_location.right(), face_location.bottom(), face_location.left()
#                     face_encodings = face_recognition.face_encodings(frame_np, [(top, right, bottom, left)])
#
#                     # Check for face match with images in the image_database
#                     for known_name, known_encoding in image_database.items():
#                         distances = face_recognition.face_distance([known_encoding], face_encodings[0])
#
#                         # Choose a suitable threshold for confidence
#                         distance_threshold = 0.5
#
#                         # Check if the distance is below the threshold
#                         if distances[0] < distance_threshold:
#                             result = {'matched': True, 'name': known_name, 'confidence': 1 - distances[0],
#                                       'message': 'Match Found!'}
#                             socketio.emit("face_recognition_result", result)
#                             logging.info("====result %s:", result)
#                             break
#                         else:
#                             result = {'matched': False, 'name': "Unknown",
#                                       'message': 'Not matching face with the DB stored image'}
#                             socketio.emit("face_recognition_result", result)
#                             logging.info("====result %s:", result)
#
#             else:
#                 socketio.emit('face_recognition_result', {'matched': False, 'message': 'No face detected'})
#
#         else:
#             socketio.emit('face_recognition_result', {'matched': False, 'message': 'Please provide a real face'})
#
#     except Exception as e:
#         socketio.emit('error', {'message': str(e)})
#         logging.error(f"Error: {e}")
#
#     finally:
#         # Mark that the anti-spoofing test is completed
#         socketio.session.pop('anti_spoofing_in_progress', None)
#
#
# # Event handler for user disconnecting
# @socketio.on('disconnect')
# def handle_disconnect():
#     logging.info("Client disconnected")
#     socketio.emit('disconnect_response', {'status': 'disconnected'})
#     socketio.session.pop('anti_spoofing_in_progress', None)
#
#
# if __name__ == '__main__':
#     # Run the application
#     socketio.run(app, host="0.0.0.0", debug=True, port=5001)
#
#     # Release the webcam
#     video_capture.release()
#     cv2.destroyAllWindows()
#     image_database.clear()