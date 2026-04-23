import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import requests
import time
import tempfile
import os
import threading

# -----------------------------
# CONFIG
# -----------------------------
BOT_TOKEN = "YOUR_NEW_BOT_TOKEN"
CHAT_ID = "YOUR_CHAT_ID"

ALERT_THRESHOLD = 0.8
COOLDOWN = 20  # seconds


# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mobilenet_model.h5", compile=False)


model = load_model()


# -----------------------------
# Telegram Alert (non-blocking)
# -----------------------------
def send_telegram_alert(image, probability):
    _, buffer = cv2.imencode(".jpg", image)

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"

    files = {"photo": ("fall.jpg", buffer.tobytes())}
    data = {
        "chat_id": CHAT_ID,
        "caption": f"🚨 Fall Detected!\nProbability: {probability:.2f}",
    }

    try:
        requests.post(url, data=data, files=files, timeout=5)
    except:
        pass


def send_alert_async(image, probability):
    threading.Thread(
        target=send_telegram_alert, args=(image, probability), daemon=True
    ).start()


# -----------------------------
# Preprocess
# -----------------------------
def preprocess_frame(frame, target_size=(256, 256)):
    frame = cv2.resize(frame, target_size)
    frame = frame.astype("float32") / 255.0
    return np.expand_dims(frame, axis=0)


# -----------------------------
# Shared Processing
# -----------------------------
def process_frame(img):
    input_frame = preprocess_frame(img)
    prediction = model.predict(input_frame, verbose=0)

    # ⚠️ Adjust if your model output differs
    probability_fall = 1 - prediction[0][0]

    is_fall = probability_fall >= 0.5

    if is_fall:
        label = f"Fall ({probability_fall:.2f})"
        color = (0, 0, 255)
    else:
        label = f"Non-Fall ({probability_fall:.2f})"
        color = (0, 255, 0)

    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    return img, probability_fall


# -----------------------------
# WebRTC Processor
# -----------------------------
class FallDetectionProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_alert_time = 0
        self.frame_count = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Frame skipping (performance boost)
        self.frame_count += 1
        if self.frame_count % 2 != 0:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        img, probability_fall = process_frame(img)

        # Alert logic
        current_time = time.time()
        if probability_fall >= ALERT_THRESHOLD:
            if current_time - self.last_alert_time > COOLDOWN:
                send_alert_async(img, probability_fall)
                self.last_alert_time = current_time

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# -----------------------------
# UI
# -----------------------------
st.markdown(
    """
    <div style="text-align:center;">
        <h1>🚨 Fall Detection AI System</h1>
        <p style="color:gray;">Real-time monitoring with AI + Telegram alerts</p>
    </div>
    <hr>
    """,
    unsafe_allow_html=True,
)

option = st.radio("Choose Input Source:", ["Live Webcam", "Upload Video"])

# -----------------------------
# Upload Video Mode
# -----------------------------
if option == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        last_alert_time = 0
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Frame skipping
            frame_count += 1
            if frame_count % 2 != 0:
                continue

            frame, probability_fall = process_frame(frame)

            # Alert logic
            current_time = time.time()
            if probability_fall >= ALERT_THRESHOLD:
                if current_time - last_alert_time > COOLDOWN:
                    send_alert_async(frame, probability_fall)
                    last_alert_time = current_time

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame, channels="RGB")

        cap.release()

# -----------------------------
# Live Webcam Mode (WebRTC)
# -----------------------------
elif option == "Live Webcam":
    webrtc_streamer(
        key="fall-detection",
        video_processor_factory=FallDetectionProcessor,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": True, "audio": False},
    )
