import time
import cv2
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from collections import deque



# Face Locators Vars
INNER_TOP = [13]
INNER_BOTTOM = [14]
INNER_LEFT = 78
INNER_RIGHT = 308
MOUTH = INNER_TOP + INNER_BOTTOM + [INNER_LEFT, INNER_RIGHT]
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
ALL = MOUTH + LEFT_EYE + RIGHT_EYE

REAL_MOUTH_WIDTH_CM = 5.0
EAR_WINDOW = deque(maxlen=60)

model_path = str(Path(__file__).parent.parent / 'models' / 'face_landmarker.task')
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
latest_landmarks = [None] # avoid local var and maybe useful for future

# Convert normalized points to pixel coordinates
def to_pixel(p, frame_width, frame_height):
    return np.array([p.x * frame_width, p.y * frame_height])

# EAR stands for Eye Aspect Ratio
def compute_ear(landmarks, frame_width, frame_height):

    p1 = to_pixel(landmarks[LEFT_EYE[0]], frame_width, frame_height)
    p2 = to_pixel(landmarks[LEFT_EYE[1]], frame_width, frame_height)
    p3 = to_pixel(landmarks[LEFT_EYE[2]], frame_width, frame_height)
    p4 = to_pixel(landmarks[LEFT_EYE[3]], frame_width, frame_height)
    p5 = to_pixel(landmarks[LEFT_EYE[4]], frame_width, frame_height)
    p6 = to_pixel(landmarks[LEFT_EYE[5]], frame_width, frame_height)

    ear_left = (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2 * np.linalg.norm(p1 - p4))

    p1 = to_pixel(landmarks[RIGHT_EYE[0]], frame_width, frame_height)
    p2 = to_pixel(landmarks[RIGHT_EYE[1]], frame_width, frame_height)
    p3 = to_pixel(landmarks[RIGHT_EYE[2]], frame_width, frame_height)
    p4 = to_pixel(landmarks[RIGHT_EYE[3]], frame_width, frame_height)
    p5 = to_pixel(landmarks[RIGHT_EYE[4]], frame_width, frame_height)
    p6 = to_pixel(landmarks[RIGHT_EYE[5]], frame_width, frame_height)

    ear_right = (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2 * np.linalg.norm(p1 - p4))

    ear = (ear_left + ear_right) / 2

    return ear

# call back func, I need to do sth here
# for now, extract the landmarks value
def print_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    if result.face_landmarks:
        latest_landmarks[0] = result.face_landmarks[0]

def main():
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_faces=1,
        result_callback=print_result)

    with FaceLandmarker.create_from_options(options) as landmarker:

        cap = cv2.VideoCapture(0)

        # read frame, convert to mp image then detect then show
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame (Hint: maybe camera is off?)")
                break

            frame = cv2.flip(frame, 1)
            frame_height = frame.shape[0]
            frame_width  = frame.shape[1]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )
            timestamp_ms = int(time.time() * 1000)
            landmarker.detect_async(mp_image, timestamp_ms)

            if latest_landmarks[0]:
                for p in ALL:
                    x = int(latest_landmarks[0][p].x * frame_width)
                    y = int(latest_landmarks[0][p].y * frame_height)
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

                ear = compute_ear(latest_landmarks[0], frame_width, frame_height)
                ear_threshold = 0.25

                EAR_WINDOW.append(ear)

                perclos = sum(1 for e in EAR_WINDOW if e < ear_threshold) / len(EAR_WINDOW)
                
                # if perclos is over 80% it is high chance the user's eyes "droops" - a classic sign of drowy
                if perclos > 0.8:
                    cv2.putText(frame, "User is drowsy", (50,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            cv2.imshow("demo", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break
            if cv2.getWindowProperty("demo", cv2.WND_PROP_VISIBLE) < 1:
                break

if __name__=="__main__":
    main()