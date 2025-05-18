import cv2
import mediapipe as mp
import numpy as np
import math

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

colour = (255, 255, 255)
width = 3
radius = 4
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.8  # Increased font size
font_color = (255, 255, 255)  # White

def draw(frame, point1, point2):
    h, w = frame.shape[:2]
    p1 = (int(point1[0] * w), int(point1[1] * h))
    p2 = (int(point2[0] * w), int(point2[1] * h))
    cv2.line(frame, p1, p2, colour, width)
    cv2.circle(frame, p1, radius, colour, cv2.FILLED)
    cv2.circle(frame, p2, radius, colour, cv2.FILLED)

def get_coords(landmarks, landmark_enum):
    lm = landmarks[landmark_enum.value]
    return [lm.x, lm.y]

def calc_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ab = a - b
    cb = c - b
    radians = np.arccos(np.clip(np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb)), -1.0, 1.0))
    return np.degrees(radians)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # Key points
        l_shoulder = get_coords(lm, mp_pose.PoseLandmark.LEFT_SHOULDER)
        l_elbow = get_coords(lm, mp_pose.PoseLandmark.LEFT_ELBOW)
        l_wrist = get_coords(lm, mp_pose.PoseLandmark.LEFT_WRIST)

        r_shoulder = get_coords(lm, mp_pose.PoseLandmark.RIGHT_SHOULDER)
        r_elbow = get_coords(lm, mp_pose.PoseLandmark.RIGHT_ELBOW)
        r_wrist = get_coords(lm, mp_pose.PoseLandmark.RIGHT_WRIST)

        l_hip = get_coords(lm, mp_pose.PoseLandmark.LEFT_HIP)
        l_knee = get_coords(lm, mp_pose.PoseLandmark.LEFT_KNEE)
        l_ankle = get_coords(lm, mp_pose.PoseLandmark.LEFT_ANKLE)

        r_hip = get_coords(lm, mp_pose.PoseLandmark.RIGHT_HIP)
        r_knee = get_coords(lm, mp_pose.PoseLandmark.RIGHT_KNEE)
        r_ankle = get_coords(lm, mp_pose.PoseLandmark.RIGHT_ANKLE)

        # Midpoints
        mid_shoulder = [(l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2]
        mid_hip = [(l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2]

        # Connections (removed thumbs)
        connections = [
            (l_shoulder, l_elbow), (l_elbow, l_wrist),
            (r_shoulder, r_elbow), (r_elbow, r_wrist),
            (l_hip, l_knee), (l_knee, l_ankle),
            (r_hip, r_knee), (r_knee, r_ankle),
            (l_shoulder, r_shoulder), (mid_shoulder, mid_hip)
        ]

        for p1, p2 in connections:
            draw(frame, p1, p2)

        # Display angles
        for a, b, c in [
            (l_shoulder, l_elbow, l_wrist),
            (r_shoulder, r_elbow, r_wrist),
            (l_hip, l_knee, l_ankle),
            (r_hip, r_knee, r_ankle)
        ]:
            angle = int(calc_angle(a, b, c))
            h, w = frame.shape[:2]
            px, py = int(b[0] * w), int(b[1] * h)
            cv2.putText(frame, f'{angle}°', (px + 10, py - 10), font, font_scale, font_color, 2)

    cv2.imshow('Body Tracker Demo', frame)
    key = cv2.waitKey(1)
    #27 = escape key | cv2.WND_PROP_VISIBLE -> whether window is visible or not (I.E. minimized)
    if key == 27 or cv2.getWindowProperty('Body Tracker Demo', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
