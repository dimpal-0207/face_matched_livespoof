# https://martlgap.medium.com/how-to-build-a-simple-live-face-recognition-app-in-python-529fc686b475
import cv2
import face_recognition
import numpy as np

# Load images from the database
image_database = {
    'dimpal': face_recognition.load_image_file('images/captured_image_20231124115603.png'),
    'joan david': face_recognition.load_image_file('images/captured_image_0.png'),
    # Add more images as needed
}

# Extract face encodings from the database images
known_face_encodings = {}
for person, image in image_database.items():
    print(person, )
    face_encoding = face_recognition.face_encodings(image)
    if face_encoding:
        known_face_encodings[person] = face_encoding[0]

# Open the camera
video_capture = cv2.VideoCapture(0)

while True:
    # Capture each frame from the camera
    ret, frame = video_capture.read()

    # Find face locations and face encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare the current face with each face in the database
        matches = face_recognition.compare_faces(list(known_face_encodings.values()), face_encoding)

        name = "Unknown"  # Default name if no match found

        # If a match is found, use the name from the database
        if True in matches:
            first_match_index = matches.index(True)
            name = list(known_face_encodings.keys())[first_match_index]

        # Draw a rectangle around the face and display the name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
video_capture.release()
cv2.destroyAllWindows()
