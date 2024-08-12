import os
import logging
import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Suppress absl warnings
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

# Initialize variables
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
screen_w, screen_h = pyautogui.size()
eye_look_count = {'right': 0, 'left': 0}
eye_away = False
last_left_look_time = time.time()
last_right_look_time = time.time()
person_count = 0
phone_detected = False

# Load the YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

def detect_phone(frame):
    global phone_detected
    phone_detected = False
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label == "cell phone":
                phone_detected = True
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (x + w // 2, y + h // 2), 5, (0, 0, 255), -1)
    return frame

def process_frame(frame):
    global eye_away
    global eye_look_count
    global last_left_look_time
    global last_right_look_time
    global person_count
    global phone_detected

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    person_count = 0

    # Face detection
    face_detection_results = face_detection.process(rgb_frame)
    if face_detection_results.detections:
        for detection in face_detection_results.detections:
            person_count += 1

    # Phone detection
    frame = detect_phone(frame)

    if landmark_points:
        landmarks = landmark_points[0].landmark
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 3, (0, 255, 0))
            if id == 1:
                screen_x = screen_w - (screen_w * landmark.x)
                screen_y = screen_h * landmark.y
                pyautogui.moveTo(screen_x, screen_y)

                # Eye look detection
        left_eye = [landmarks[145], landmarks[159]]
        right_eye = [landmarks[362], landmarks[386]]
        left_pupil = landmarks[468]
        right_pupil = landmarks[473]
        for landmark in left_eye:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 3, (0, 255, 255))
        for landmark in right_eye:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 3, (255, 0, 0))

        # Count eye looks
        current_time = time.time()
        if left_pupil.x < left_eye[0].x and current_time - last_left_look_time > 0.5:
            eye_look_count['left'] += 1
            last_left_look_time = current_time
        if right_pupil.x > right_eye[0].x and current_time - last_right_look_time > 0.5:
            eye_look_count['right'] += 1
            last_right_look_time = current_time

        # Detect if eyes are away from screen
        left_eye_center_x = (left_eye[0].x + left_eye[1].x) / 2
        right_eye_center_x = (right_eye[0].x + right_eye[1].x) / 2
        left_eye_distance = abs(left_pupil.x - left_eye_center_x)
        right_eye_distance = abs(right_pupil.x - right_eye_center_x)

        if left_eye_distance > 0.01 or right_eye_distance > 0.01:
            eye_away = True
        else:
            eye_away = False

    # Display eye look counts and away status
    cv2.rectangle(frame, (0, frame.shape[0] - 100), (frame.shape[1], frame.shape[0]), (255, 255, 255), -1)
    cv2.putText(frame, f'Left: {eye_look_count["left"]}', (10, frame.shape[0] - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 139), 2)
    cv2.putText(frame, f'Right: {eye_look_count["right"]}', (150, frame.shape[0] - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 139), 2)
    cv2.putText(frame, f'Away: {eye_away}', (300, frame.shape[0] - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 139), 2)
    cv2.putText(frame, f'Person Count: {person_count}', (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 139), 2)
    cv2.putText(frame, f'Phone Detected: {phone_detected}', (200, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 139), 2)
    cv2.putText(frame, "Press ESC to quit", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return frame

while True:
    _, frame = cam.read()
    frame = process_frame(frame)
    cv2.imshow('Eye Controlled Mouse', frame)
    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        break