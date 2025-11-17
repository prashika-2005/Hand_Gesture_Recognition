Hand Gesture Recognition (A–Z)
Using MediaPipe + Random Forest + Flask + Real-Time Audio Feedback

Authors:

Prashika S. Lonkar

Sharayu S. Madage

Samruddhi Mane

Institution:
MKSSS Cummins College of Engineering for Women, Pune, India

-> Overview

This project implements a real-time hand gesture recognition system capable of predicting static hand gestures for A–Z alphabets using:

MediaPipe Hands for extracting 21 hand landmarks

Random Forest Classifier for gesture prediction

Flask web server for real-time video streaming

gTTS + playsound for instant audio feedback (“Letter A”, “Letter B”, …)

This system can be used for:

Sign language learning

Accessibility applications

Human–computer interaction (HCI)

Educational tools and demos

-> Project Structure
├── README.md
├── requirements.txt
├── app.py                     # Flask application for real-time inference
├── collect_dataset.py         # Capture images for dataset (A–Z)
├── extract_landmarks.py       # MediaPipe landmark extraction + save data.pickle
├── train_model.py             # Train RandomForest model + save model.p
├── model.p                    # Trained model
├── data.pickle                # Extracted features + labels
├── data/                      # Dataset folder (A/, B/, C/, ...)
├── sounds/                    # Audio files (A.mp3 ... Z.mp3)
├── templates/
│   └── index.html             # Web UI for video streaming
└── static/
    └── css/ (optional)

-> Installation
1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/macOS

2. Install dependencies
pip install -r requirements.txt

3. Example requirements.txt
opencv-python
mediapipe
numpy
flask
scikit-learn
gTTS
playsound==1.2.2
joblib
pillow

-> Dataset Collection (A–Z)

Run this script to capture 100 images for each alphabet gesture:

python collect_dataset.py

Controls:

Press c → capture image

Press q → go to next letter

Press Esc → exit

All images are stored in:

data/<LETTER>/


Example: data/A/0.jpg

-> Feature Extraction (MediaPipe Landmarks)

Extract 21 (x, y) hand landmarks per image:

python extract_landmarks.py


The script:

Reads images from /data

Uses MediaPipe Hands

Normalizes features:

x'i = xi – min(x)

y'i = yi – min(y)

Saves dataset as data.pickle

-> Model Training (Random Forest)

Train the classifier:

python train_model.py


Model details:

Random Forest (200 trees, depth=20)

Train/Test = 80/20 split

Accuracy: 95–98%

Saved as:

model.p

Run Real-Time Flask App

Start the server:

python app.py


Open your browser:

http://127.0.0.1:5000/

->Features:

Start/stop webcam

Real-time prediction

Landmark overlays

Audio announcement of recognized letter

API Endpoints:

GET  /               → index page
POST /start_camera   → start webcam
POST /stop_camera    → stop webcam
GET  /video_feed     → live video stream
GET  /get_gesture    → returns last predicted gesture
POST /exit           → shutdown server

-> Audio Feedback (A–Z)

Generate MP3 files for each alphabet using:

make_sounds.py


Script:
from gtts import gTTS
import os

os.makedirs("sounds", exist_ok=True)

for ch in range(ord('A'), ord('Z') + 1):
    letter = chr(ch)
    tts = gTTS(text=f"Letter {letter}", lang='en')
    tts.save(f"sounds/{letter}.mp3")

📈 Results
Metric	Value
Training Accuracy	98%
Testing Accuracy	96%
Dataset Size	2600 images
Features	42 landmarks
Prediction Latency	< 0.02 sec
Real-Time FPS	30+

Your accuracy graph (accuracy_chart.png) can be added to the repo.

🧪 Troubleshooting
❗ WebCam Not Starting

Check if other apps are using the camera.

❗ playsound Issue

Windows: works fine
Linux: use python-vlc instead

❗ Low Accuracy for M/N

Add more training samples
Keep background simple
Improve lighting conditions

📚 References
Lugaresi et al., “MediaPipe: A Framework for Building Perception Pipelines,” arXiv, 2019

Breiman, “Random Forests,” Machine Learning, 2001

Camgoz et al., “SubUNet,” ICCV, 2017

Molchanov et al., “3D CNN for Gesture Recognition,” CVPR, 2016


📧 Contact

For help or support:

📩 prashika.lonkar@cumminscollege.in
