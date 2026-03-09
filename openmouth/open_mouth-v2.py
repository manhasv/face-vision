import time
import cv2
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

INNER_TOP = [13]
INNER_BOTTOM = [14]
INNER_LEFT = 78
INNER_RIGHT = 308
ALL = INNER_TOP + INNER_BOTTOM + [INNER_LEFT, INNER_RIGHT]

REAL_MOUTH_WIDTH_CM = 5.0

model_path = str(Path(__file__).parent.parent / 'models' / 'face_landmarker.task')

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
latest_landmarks = [None] # avoid local var and maybe useful for future

def compute_mar(landmarks, frame_width, frame_height):
    # Convert normalized points to pixel coordinates
    def to_pixel(p):
        return np.array([p.x * frame_width, p.y * frame_height])

    # Average top and bottom lip y positions
    top_y = np.mean([to_pixel(landmarks[i])[1] for i in INNER_TOP])
    bottom_y = np.mean([to_pixel(landmarks[i])[1] for i in INNER_BOTTOM])

    # Mouth vertical distance
    vertical = bottom_y - top_y

    # Horizontal distance (corners)
    left = to_pixel(landmarks[INNER_LEFT])
    right = to_pixel(landmarks[INNER_RIGHT])
    horizontal = np.linalg.norm(right - left)

    # MAR = vertical / horizontal
    mar = vertical / horizontal
    return mar

def estimate_distance(landmarks, frame_width, frame_height, focal_length):
    left = np.array([landmarks[INNER_LEFT].x * frame_width, landmarks[INNER_LEFT].y * frame_height])
    right = np.array([landmarks[INNER_RIGHT].x * frame_width, landmarks[INNER_RIGHT].y * frame_height])
    pixel_width = np.linalg.norm(right - left)

    if pixel_width == 0:
        return None  # avoid division by zero
    distance_cm = REAL_MOUTH_WIDTH_CM * focal_length / pixel_width
    return distance_cm

def estimate_relative_distance(landmarks):
    # Average z of inner lips (negative z = closer)
    z_values = [landmarks[i].z for i in ALL]
    avg_z = np.mean(z_values)
    # Flip sign so closer faces are higher positive value (optional)
    return -avg_z

# call back func, do sth here
# for now, extract the landmarks value
def print_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    if result.face_landmarks:
        latest_landmarks[0] = result.face_landmarks[0]

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_faces=1,
    result_callback=print_result)

with FaceLandmarker.create_from_options(options) as landmarker:

    cap = cv2.VideoCapture(0)

    # while true, read frame, convert to mp image then detect then show
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
            mar = compute_mar(latest_landmarks[0], frame_width, frame_height)
            threshold = 0.3
            if mar > threshold:
                cv2.putText(frame, "OPEN", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            else:
                cv2.putText(frame, "CLOSED", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
            # Rough relative distance using z-coordinates
            rel_distance = 20 * estimate_relative_distance(latest_landmarks[0])
            cv2.putText(frame, f"Rel Dist: {rel_distance:.3f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


        cv2.imshow("demo", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break
        if cv2.getWindowProperty("demo", cv2.WND_PROP_VISIBLE) < 1:
            break