import cv2
from PIL import ImageTk, Image

from test import test
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)  # Use 0 for webcam or provide the path to a video file

while True:
    # Read the current frame
    ret, frame = video_capture.read()
    most_recent_capture_array = frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    most_recent_capture_arr = frame
    img_ = cv2.cvtColor(most_recent_capture_arr, cv2.COLOR_BGR2RGB)
    most_recent_capture_pil = Image.fromarray(img_)
    # imgtk = ImageTk.PhotoImage(image=most_recent_capture_pil)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    print("====length", len(faces))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # If no faces are detected, output fake detection
    if len(faces) == 1:
        # Perform further actions for fake detection

        label, value = test(image=most_recent_capture_array,
                            model_dir=r"C:\Users\Admin\Downloads\Silent-Face-Anti-Spoofing-master\Silent-Face-Anti-Spoofing-master\resources\anti_spoof_models",
                            device_id=0)
    # Process each face detected
    # for (x, y, w, h) in faces:
    #     # Extract the face region of interest (ROI)
    #     face_roi = frame[y:y + h, x:x + w]
    #     gray_roi = gray[y:y + h, x:x + w]
        if label == 1:
            print("Image '{}' is Real Face. Score: {:.2f}.")
            result_text = "RealFace Score: {:.2f}".format(value)
            color = (255, 0, 0)
        else:
            print("Image '{}' is Fake Face. Score: {:.2f}.")
            result_text = "FakeFace Score: {:.2f}".format(value)
            color = (0, 0, 255)
    else:
        print("Image '{}' is Fake Face. Score: {:.2f}.")
        result_text = "FakeFace Score: {:.2f}"
        color = (0, 0, 255)

    # else:
    #     # util.msg_box('Hey, you are a spoofer!', 'You are fake !')
    #     print("hey you are spoofer ")

    # Display the resulting frame
    cv2.imshow('Liveness Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break