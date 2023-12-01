# import cv2
# import os
#

import cv2
import os
from datetime import datetime

# Set the directory where images will be saved
save_directory = r"C:\Users\Admin\PycharmProjects\Facedetection_matched_ML\images"  # Replace with the actual path

# Create the save directory if it doesn't exist
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Open the webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture each frame from the camera
    ret, frame = video_capture.read()

    # Display the frame in a window
    cv2.imshow('Capture Images', frame)

    # Check for the 's' key to save the image
    key = cv2.waitKey(1)
    if key == ord('s'):
        # Generate a unique filename using a timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        image_filename = os.path.join(save_directory, f"captured_image_{timestamp}.png")

        # Save the image to the specified directory
        cv2.imwrite(image_filename, frame)
        print(f"Image saved: {image_filename}")

    # Break the loop when 'q' key is pressed
    elif key == ord('q'):
        break

# Release the webcam and close all windows
video_capture.release()
cv2.destroyAllWindows()

# # Set the directory where images will be saved
# save_directory = r"C:\Users\Admin\PycharmProjects\Facedetection_matched_ML/images"  # Replace with the actual path
#
# # Create the save directory if it doesn't exist
# if not os.path.exists(save_directory):
#     os.makedirs(save_directory)
#
# # Open the webcam
# video_capture = cv2.VideoCapture(0)
#
# # Counter for image filenames
# image_count = 0
#
# while True:
#     # Capture each frame from the camera
#     ret, frame = video_capture.read()
#
#     # Display the frame in a window
#     cv2.imshow('Capture Images', frame)
#
#     # Check for the 's' key to save the image
#     key = cv2.waitKey(1)
#     if key == ord('s'):
#         # Save the image to the specified directory
#         image_filename = os.path.join(save_directory, f"captured_image_{image_count}.png")
#         cv2.imwrite(image_filename, frame)
#         print(f"Image saved: {image_filename}")
#         image_count += 1
#
#     # Break the loop when 'q' key is pressed
#     elif key == ord('q'):
#         break
#
# # Release the webcam and close all windows
# video_capture.release()
# cv2.destroyAllWindows()
