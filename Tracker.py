import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose and Face Mesh
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Drawing specs for white lines and dots
colour = (255, 255, 255)
width = 2
radius = 2

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

cap = cv2.VideoCapture(0)

# Select first 300 points for reduced detail face mesh
subset_indices = set(range(0, 350))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pose_results = pose.process(rgb_frame)
    face_results = face_mesh.process(rgb_frame)

    # --- Draw Pose Landmarks ---
    if pose_results.pose_landmarks:
        lm = pose_results.pose_landmarks.landmark

        left_shoulder = get_coords(lm, mp_pose.PoseLandmark.LEFT_SHOULDER)
        left_elbow = get_coords(lm, mp_pose.PoseLandmark.LEFT_ELBOW)
        left_wrist = get_coords(lm, mp_pose.PoseLandmark.LEFT_WRIST)

        right_shoulder = get_coords(lm, mp_pose.PoseLandmark.RIGHT_SHOULDER)
        right_elbow = get_coords(lm, mp_pose.PoseLandmark.RIGHT_ELBOW)
        right_wrist = get_coords(lm, mp_pose.PoseLandmark.RIGHT_WRIST)

        left_hip = get_coords(lm, mp_pose.PoseLandmark.LEFT_HIP)
        left_knee = get_coords(lm, mp_pose.PoseLandmark.LEFT_KNEE)
        left_ankle = get_coords(lm, mp_pose.PoseLandmark.LEFT_ANKLE)

        right_hip = get_coords(lm, mp_pose.PoseLandmark.RIGHT_HIP)
        right_knee = get_coords(lm, mp_pose.PoseLandmark.RIGHT_KNEE)
        right_ankle = get_coords(lm, mp_pose.PoseLandmark.RIGHT_ANKLE)

        pose_connections = [
            (left_shoulder, left_elbow), (left_elbow, left_wrist),
            (right_shoulder, right_elbow), (right_elbow, right_wrist),
            (left_hip, left_knee), (left_knee, left_ankle),
            (right_hip, right_knee), (right_knee, right_ankle),
            (left_shoulder, right_shoulder),  # shoulder line
            (left_hip, right_hip)
        ]

        for p1, p2 in pose_connections:
            draw(frame, p1, p2)

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            for connection in mp_face_mesh.FACEMESH_TESSELATION:
                start_idx, end_idx = connection
                if start_idx in subset_indices and end_idx in subset_indices:
                    start = face_landmarks.landmark[start_idx]
                    end = face_landmarks.landmark[end_idx]
                    draw(frame, (start.x, start.y), (end.x, end.y))

    cv2.imshow('Body and Face Tracking', frame)
    key = cv2.waitKey(1)

    if key == 27:  # ESC key
        break

    prop = cv2.getWindowProperty('Body and Face Tracking', cv2.WND_PROP_VISIBLE)
    if prop < 1:
        break

cap.release()
cv2.destroyAllWindows()
