import cv2
import dlib
import math
from pathlib import Path

# **This script uses dlib instead of Mediapipe Face Landmarker**

# The detector finds the face rectangle
detector = dlib.get_frontal_face_detector()
# The predictor finds the 68 specific points inside the face rect
predictor = dlib.shape_predictor(str(Path(__file__).parent.parent / 'models' / 'shape_predictor_68_face_landmarks.dat'))

def calculate_distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

cap = cv2.VideoCapture(0)

while True:
    # 'ret' is a boolean (True/False) indicating if the read was successful
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame (Hint: maybe camera is off?). Exiting ...")
        break

    # convert to grayscale for easier detection (i think? I read somewhere that this should be the standard, might need more research)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        #Get Key Mouth Points (Dlib 68-point model)
        # Top lip: 51, Bottom lip: 57
        top_lip = landmarks.part(51)
        bottom_lip = landmarks.part(57)
        
        # Left corner: 48, Right corner: 54
        left_corner = landmarks.part(48)
        right_corner = landmarks.part(54)

        # Draw circle on landmarks points
        for p in [top_lip, bottom_lip, left_corner, right_corner]:
            cv2.circle(frame, (p.x, p.y), 3, (0, 255, 0), -1)

        # Logic for calc here
        vertical_dist = calculate_distance(top_lip, bottom_lip)
        horizontal_dist = calculate_distance(left_corner, right_corner)

        mar = vertical_dist / horizontal_dist if horizontal_dist > 0 else 0

        if mar > 0.6:
            cv2.putText(frame, "OPEN", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "CLOSED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Dlib Mouth Detector", frame)
    # Esc or close window will stop
    if cv2.waitKey(1) & 0xFF == 27:
        break
    if cv2.getWindowProperty("Dlib Mouth Detector", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()