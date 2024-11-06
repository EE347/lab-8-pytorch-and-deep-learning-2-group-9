import cv2
from picamera2 import Picamera2, previews
import numpy as np
import time
import os

# Ensure the directories exist
train_dir = 'data/train/0'
test_dir = 'data/test/0'

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the camera
camera = Picamera2()
camera.configure(camera.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
camera.start()
time.sleep(0.1)

# Initialize counters
train_count = 0
test_count = 0
max_train_photos = 50
max_test_photos = 10

# Start capturing frames
while True:
    frame = camera.capture_array()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for face detection
    cv2.imshow("task9", frame_gray)

    # Wait for spacebar key to start taking 60 photos

    if cv2.waitKey(1) & 0xFF == ord('c'):
        train_dir = 'data/train/1'
        test_dir = 'data/test/1'
        train_count=0
        test_count=0
        print('changed dir')

    if cv2.waitKey(1) & 0xFF == ord(' '):
        print("Starting to take 60 photos...")

        # Capture and save 60 different frames
        for i in range(max_train_photos):
            frame = camera.capture_array()  # Capture a new frame for each photo
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # If faces are detected, crop and save the first face found
            if len(faces) > 0:
                x, y, w, h = faces[0]  # Get the first face found
                face = frame[y:y+h, x:x+w]  # Crop the face region
                face_resized = cv2.resize(face, (64, 64))  # Resize the face to 64x64
                cv2.imwrite(os.path.join(train_dir, f"train_img_{train_count}.jpg"), face_resized)
                print(f"Saving training image {train_count}")
                train_count += 1
            time.sleep(0.1)  # Give a little delay between images

        for i in range(max_test_photos):
            frame = camera.capture_array()  # Capture a new frame for each photo
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # If faces are detected, crop and save the first face found
            if len(faces) > 0:
                x, y, w, h = faces[0]  # Get the first face found
                face = frame[y:y+h, x:x+w]  # Crop the face region
                face_resized = cv2.resize(face, (64, 64))  # Resize the face to 64x64
                cv2.imwrite(os.path.join(test_dir, f"test_img_{test_count}.jpg"), face_resized)
                print(f"Saving testing image {test_count}")
                test_count += 1
            time.sleep(0.1)  # Give a little delay between images

        print("60 photos taken, press 'q' to quit.")
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.stop()
cv2.destroyAllWindows()
