from typing import NamedTuple

import cv2
import mediapipe as mp
import numpy as np
import time  # Import time for timestamp

#import webapp

UpAngle = 160
DownAngle = 70
MinKneeAngle = 130

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

push_up_count = 0
stage = "up"

warning_start = 0
wduration = 2  # seconds

def calc_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

def draw(frame, point1, point2):
    h, w = frame.shape[:2]
    p1 = (int(point1[0] * w), int(point1[1] * h))
    p2 = (int(point2[0] * w), int(point2[1] * h))
    cv2.line(frame, p1, p2, (255, 255, 255), 3)
    cv2.circle(frame, p1, 7, (255, 255, 255), cv2.FILLED)
    cv2.circle(frame, p2, 7, (255, 255, 255), cv2.FILLED)

def get_coords(landmarks, landmark_enum):
    lm = landmarks[landmark_enum.value]
    return [lm.x, lm.y]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_height, frame_width = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results: NamedTuple = pose.process(rgb_frame)

    # ... (rest of your code above remains unchanged)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # Extract landmarks
        ls = get_coords(lm, mp_pose.PoseLandmark.LEFT_SHOULDER)
        le = get_coords(lm, mp_pose.PoseLandmark.LEFT_ELBOW)
        lw = get_coords(lm, mp_pose.PoseLandmark.LEFT_WRIST)

        rs = get_coords(lm, mp_pose.PoseLandmark.RIGHT_SHOULDER)
        re = get_coords(lm, mp_pose.PoseLandmark.RIGHT_ELBOW)
        rw = get_coords(lm, mp_pose.PoseLandmark.RIGHT_WRIST)

        lh = get_coords(lm, mp_pose.PoseLandmark.LEFT_HIP)
        lk = get_coords(lm, mp_pose.PoseLandmark.LEFT_KNEE)
        la = get_coords(lm, mp_pose.PoseLandmark.LEFT_ANKLE)

        rh = get_coords(lm, mp_pose.PoseLandmark.RIGHT_HIP)
        rk = get_coords(lm, mp_pose.PoseLandmark.RIGHT_KNEE)
        ra = get_coords(lm, mp_pose.PoseLandmark.RIGHT_ANKLE)

        lefta_ang = calc_angle(ls, le, lw)
        righta_angle = calc_angle(rs, re, rw)

        angle_lk = calc_angle(lh, lk, la)
        angle_rk = calc_angle(rh, rk, ra)

        for (p1, p2) in [
            (ls, le), (le, lw),
            (rs, re), (re, rw),
            (lh, lk), (lk, la),
            (rh, rk), (rk, ra),
            (ls, rs)]:
            draw(frame, p1, p2)

        mid_shoulder = [(ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2]
        mid_hip = [(lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2]
        draw(frame, mid_shoulder, mid_hip)

        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 1
        thickness = 2
        line_type = cv2.LINE_AA
        color = (255, 255, 255)  # White color for angle text


        # Convert normalized coords to pixel coords for text placement
        def to_pixel_coords(point):
            return int(point[0] * frame_width), int(point[1] * frame_height)


        # Draw angles next to relevant joints (with slight offset)
        offsets = {
            'shoulder': (10, -10),
            'elbow': (10, -10),
            'wrist': (10, -10),
            'knee': (10, -10),
            'ankle': (10, -10)
        }

        # Left arm angles (at elbow)
        le_px = to_pixel_coords(le)
        cv2.putText(frame, f'{int(lefta_ang)}', (le_px[0] + offsets['elbow'][0], le_px[1] + offsets['elbow'][1]),
                    font, font_scale, color, thickness, line_type)

        # Right arm angles (at elbow)
        re_px = to_pixel_coords(re)
        cv2.putText(frame, f'{int(righta_angle)}', (re_px[0] + offsets['elbow'][0], re_px[1] + offsets['elbow'][1]),
                    font, font_scale, color, thickness, line_type)

        # Left knee angle (at knee)
        lk_px = to_pixel_coords(lk)
        cv2.putText(frame, f'{int(angle_lk)}', (lk_px[0] + offsets['knee'][0], lk_px[1] + offsets['knee'][1]),
                    font, font_scale, color, thickness, line_type)

        # Right knee angle (at knee)
        rk_px = to_pixel_coords(rk)
        cv2.putText(frame, f'{int(angle_rk)}', (rk_px[0] + offsets['knee'][0], rk_px[1] + offsets['knee'][1]),
                    font, font_scale, color, thickness, line_type)

        legs_straight = angle_lk > MinKneeAngle and angle_rk > MinKneeAngle

        if lefta_ang > UpAngle and righta_angle > UpAngle:
            stage = "up"

        if lefta_ang < DownAngle and righta_angle < DownAngle and stage == "up" and legs_straight:
            stage = "down"
            push_up_count += 1
            print(f"Pushup Count: {push_up_count}")

        if not legs_straight:
            warning_start = time.time()

        # Draw black box in the top-left corner (e.g., 300x100 px)
        cv2.rectangle(frame, (20, 20), (220, 60), (0, 0, 0), thickness=cv2.FILLED)

        # Move push-up count and warning up to avoid overlapping the joint text
        cv2.putText(frame, f'Pushups: {push_up_count}', (30, 50), font, 1, (255, 255, 255), 2, line_type)

        if time.time() - warning_start < wduration:
            cv2.putText(frame, 'Keep your legs straight!', (30, 90), font, 1.5, (0, 0, 255), 3, line_type)

        calories = (push_up_count * 2) / 5
        calorie_text = f'Calories: {calories:.1f}'

        # Fixed box size and position (width=200, height=40)
        box_width, box_height = 230, 40
        box_x1 = frame_width - box_width - 30  # 30 px from right edge
        box_y1 = 10  # 10 px from top edge
        box_x2 = box_x1 + box_width
        box_y2 = box_y1 + box_height

        # Draw filled black rectangle as background
        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), thickness=cv2.FILLED)

        # Draw calorie text inside the box with some padding
        text_x = box_x1 + 10
        text_y = box_y1 + 30  # vertical position for text baseline

        cv2.putText(frame, calorie_text, (text_x, text_y), font, font_scale, (255, 255, 0), thickness, line_type)

    cv2.imshow('PushUp Counter', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

#webapp.displayStatistics("pushup",push_up_count)
cv2.destroyAllWindows()
