import cv2
import numpy as np
import face_recognition
import boto3
from boto3.resources import response
from botocore.exceptions import NoCredentialsError

# AWS S3 credentials
AWS_ACCESS_KEY = 'AKIAVKN6BJKO6RJCCXMQ'
AWS_SECRET_KEY = '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'
BUCKET_NAME = 'facedettest'

FOLDER_NAME = 'images/'

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

        # Extract the name from the object key (you may need to customize this based on your naming conventions)
        face_name = object_key.split('/')[-1].split('.')[0]

        # Add face encoding to the image_database
        image_database[face_name] = face_encoding


import cv2
import face_recognition
import numpy as np



# OpenCV setup for live face detection
video_capture = cv2.VideoCapture(0)

while True:
    # Capture each frame from the camera
    ret, frame = video_capture.read()

    # Convert BGR to RGB (OpenCV uses BGR by default, but face_recognition uses RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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

        # Draw a rectangle around the face and display the name
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('Live Face Recognition', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows

video_capture.release()
cv2.destroyAllWindows()


