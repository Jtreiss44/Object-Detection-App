import streamlit as st
from vidgear.gears import CamGear
from ultralytics import YOLO
import cv2
from datetime import datetime
import os

st.set_page_config(layout="wide")
st.title("🚂 Live Train Detection")

youtube_url = st.text_input(
    "YouTube Stream URL",
    "https://www.youtube.com/watch?v=xKUkjFJkKgc"
)

start = st.button("Start Stream")

frame_placeholder = st.empty()

if start:

    model = YOLO("yolov8n.pt")

    stream = CamGear(
        source=youtube_url,
        stream_mode=True,
        logging=False
    ).start()

    train_present = False
    consecutive_frames = 0
    REQUIRED_FRAMES = 3
    frame_count = 0

    while True:
        frame = stream.read()
        if frame is None:
            st.warning("Stream ended.")
            break

        frame_count += 1

        # Process every 3rd frame for speed
        if frame_count % 3 != 0:
            continue

        h, w, _ = frame.shape

        # Center ROI
        roi_w = int(w * 0.5)
        roi_h = int(h * 0.3)

        x_start = (w - roi_w) // 2
        y_start = (h - roi_h) // 2
        x_end = x_start + roi_w
        y_end = y_start + roi_h

        center_roi = frame[y_start:y_end, x_start:x_end]

        results = model(center_roi, imgsz=640, conf=0.5)

        detected_this_frame = False

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]

                if label == "train":
                    detected_this_frame = True

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Shift back to original frame
                    x1 += x_start
                    x2 += x_start
                    y1 += y_start
                    y2 += y_start

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "TRAIN",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 255, 0), 2)

        # Stability logic
        if detected_this_frame:
            consecutive_frames += 1
        else:
            consecutive_frames = 0

        if consecutive_frames >= REQUIRED_FRAMES and not train_present:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            st.success(f"Train detected at {timestamp}")
            train_present = True

        if consecutive_frames == 0:
            train_present = False

        # Draw ROI guide
        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)

        # Convert BGR to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display frame on webpage
        frame_placeholder.image(frame_rgb, channels="RGB")

    stream.stop()
