# Project requirement

## Dependencies

Install Python packages:

```
pip install -r requirements.txt
```

## Model

This project uses MediaPipe's Face Landmarker model. Download it and place it at `models/face_landmarker.task` (one level above this folder):

- Model: `face_landmarker.task`
- Download: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker

## Running

Activate the virtual environment:

**Windows**
```
.venv\Scripts\activate
```

**Mac/Linux**
```
source .venv/bin/activate
```

Then run:
```
python drowsy/v3.py
```

Press `Esc` or close the window to quit.