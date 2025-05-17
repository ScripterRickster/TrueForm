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

def draw_line_and_dots(frame, point1, point2):
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

        # Key landmarks
        ls = get_coords(lm, mp_pose.PoseLandmark.LEFT_SHOULDER)
        rs = get_coords(lm, mp_pose.PoseLandmark.RIGHT_SHOULDER)
        lh = get_coords(lm, mp_pose.PoseLandmark.LEFT_HIP)
        rh = get_coords(lm, mp_pose.PoseLandmark.RIGHT_HIP)
        lk = get_coords(lm, mp_pose.PoseLandmark.LEFT_KNEE)
        rk = get_coords(lm, mp_pose.PoseLandmark.RIGHT_KNEE)

        # Midpoints
        ms = [(ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2]
        mh = [(lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2]
        mk = [(lk[0] + rk[0]) / 2, (lk[1] + rk[1]) / 2]

        # Torso angle: knee - hip - shoulder
        torso = calc_angle(mk, mh, ms)

        # Draw skeleton lines
        draw_line_and_dots(frame, mk, mh)
        draw_line_and_dots(frame, mh, ms)
        draw_line_and_dots(frame, ls, rs)
        draw_line_and_dots(frame, lh, rh)

        # Sit-up logic
        if torso < UpAngle and stage == "down":
            stage = "up"
        elif torso > DownAngle and stage == "up":
            stage = "down"
            count += 1
            calories += 0.5
            print(f"Sit-up Count: {count}")

        # UI overlays
        font, font_scale, thickness = cv2.FONT_HERSHEY_DUPLEX, 1, 2
        line_type = cv2.LINE_AA

        # Text content
        torso_text = f'Torso Angle: {int(torso)}'
        count_text = f'Sit-ups: {count}'

        # Calculate text sizes
        (tw1, th1), _ = cv2.getTextSize(torso_text, font, font_scale, thickness)
        (tw2, th2), _ = cv2.getTextSize(count_text, font, 1, 2)

        # Determine box dimensions
        box_w = max(tw1, tw2) + 20
        box_h = th1 + th2 + 50
        box_x, box_y = 20, 20

        # Draw background rectangle
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), thickness=-1)

        # Draw text over the rectangle
        cv2.putText(frame, torso_text, (box_x + 10, box_y + th1 + 10), font, font_scale, (0, 255, 0), thickness,
                    line_type)
        cv2.putText(frame, count_text, (box_x + 10, box_y + th1 + th2 + 35), font, 1, (0, 255, 255), 2, line_type)

        # Calorie Counter - Top right corner
        # Calorie Counter - Top right corner with background rectangle
        text = f'Calories: {calories:.1f}'
        (text_w, text_h), _ = cv2.getTextSize(text, font, 0.9, 2)

        # Box position and dimensions
        box_width = 230
        box_height = 40
        margin = 20
        pos_x = frame.shape[1] - box_width - margin
        pos_y = margin

        # Draw rectangle background
        cv2.rectangle(frame, (pos_x, pos_y - 10), (pos_x + box_width, pos_y + box_height), (0, 0, 0), thickness=-1)

        # Draw text over rectangle (vertically centered)
        text_x = pos_x + 10
        text_y = pos_y + int((box_height + text_h) / 2) - 5
        cv2.putText(frame, text, (text_x, text_y), font, 1, (255, 255, 0), 2, line_type)

    cv2.imshow("Sit-Up Counter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
