import cv2
import mediapipe as mp
import numpy as np

UpAngle = 45
DownAngle = 120

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)

count = 0
calories = 0.0
stage = "down"

def calc_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    ang = np.abs(rad * 180.0 / np.pi)
    return 360 - ang if ang > 180 else ang

def drawline(frame, point1, point2):
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
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        ls = get_coords(lm, mp_pose.PoseLandmark.LEFT_SHOULDER)
        rs = get_coords(lm, mp_pose.PoseLandmark.RIGHT_SHOULDER)
        lh = get_coords(lm, mp_pose.PoseLandmark.LEFT_HIP)
        rh = get_coords(lm, mp_pose.PoseLandmark.RIGHT_HIP)
        lk = get_coords(lm, mp_pose.PoseLandmark.LEFT_KNEE)
        rk = get_coords(lm, mp_pose.PoseLandmark.RIGHT_KNEE)

        ms = [(ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2]
        mh = [(lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2]
        mk = [(lk[0] + rk[0]) / 2, (lk[1] + rk[1]) / 2]

        torso = calc_angle(mk, mh, ms)

        drawline(frame, mk, mh)
        drawline(frame, mh, ms)
        drawline(frame, ls, rs)
        drawline(frame, lh, rh)

        if torso < UpAngle and stage == "down":
            stage = "up"
        elif torso > DownAngle and stage == "up":
            stage = "down"
            count += 1
            calories += 0.5
            print(f"Sit-up Count: {count}")

        font, font_scale, thickness = cv2.FONT_HERSHEY_DUPLEX, 1, 2
        line_type = cv2.LINE_AA

        torso_text = f'Torso Angle: {int(torso)}'
        count_text = f'Sit-ups: {count}'
        text = f'Calories: {calories:.1f}'

        (tw1, th1), _ = cv2.getTextSize(torso_text, font, font_scale, thickness)
        (tw2, th2), _ = cv2.getTextSize(count_text, font, 1, 2)
        (text_w, text_h), _ = cv2.getTextSize(text, font, 0.9, 2)

        boxaw = max(tw1, tw2) + 20
        boxah = th1 + th2 + 50
        boxx, boxy = 20, 20

        cv2.rectangle(frame, (boxx, boxy), (boxx + boxaw, boxy + boxah), (0, 0, 0), thickness=-1)
        cv2.putText(frame, torso_text, (boxx + 10, boxy + th1 + 10), font, font_scale, (0, 255, 0), thickness, line_type)
        cv2.putText(frame, count_text, (boxx + 10, boxy + th1 + th2 + 35), font, 1, (0, 255, 255), 2, line_type)

        boxbw = 230
        boxbh = 40
        margin = 20
        posx = frame.shape[1] - boxbw - margin
        posy = margin

        cv2.rectangle(frame, (posx, posy - 10), (posx + boxbw, posy + boxbh), (0, 0, 0), thickness=-1)
        text_x = posx + 10
        text_y = posy + int((boxbh + text_h) / 2) - 5
        cv2.putText(frame, text, (text_x, text_y), font, 1, (255, 255, 0), 2, line_type)

    cv2.imshow('Sit-Up Counter', frame)
    key = cv2.waitKey(1)

    if key == 27:  # ESC key
        break

    prop = cv2.getWindowProperty('Sit-Up Counter', cv2.WND_PROP_VISIBLE)
    if prop < 1:
        break

cap.release()
cv2.destroyAllWindows()
