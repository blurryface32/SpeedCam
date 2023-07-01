import streamlit as st
import cv2
import numpy as np
from object_detection import ObjectDetection
from speed_calc import EuclideanDistTracker

# Initialize Object Detection
od = ObjectDetection()

# Create Tracker Object
tracker = EuclideanDistTracker()

# Define Streamlit app layout
st.title("Speed Cam")
video_file = st.file_uploader("Upload a video file", type=["mp4"])

# Function to process the video
def process_video(video_file, distance, pixel):
    # Convert the file-like object to a OpenCV VideoCapture object
    video_bytes = video_file.read()
    video_nparray = np.frombuffer(video_bytes, np.uint8)
    frame = cv2.imdecode(video_nparray, cv2.IMREAD_COLOR)
    cap = cv2.VideoCapture()
    cap.open("output.mp4")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (frame_width, frame_height))

    ratio = distance / pixel
    ht = frame_height

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Object detection
        detections = od.detect(frame)

        # Speed calculation and drawing
        objects_bbs_ids = tracker.update(detections, ht)
        for obj_bb_id in objects_bbs_ids:
            x, y, w, h, obj_id = obj_bb_id
            dist = ratio * (0.375 * ht)
            speed = tracker.get_speed(obj_id, dist)
            cv2.putText(frame, str(speed), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Display the frame
        cv2.imshow("Speed Cam", frame)
        out.write(frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Process the uploaded video file
if video_file is not None:
    distance = st.number_input("Enter the distance:", value=368)
    pixel = st.number_input("Enter the video quality:", value=1080)

    st.write("Processing video...")
    process_video(video_file, distance, pixel)
    st.write("Video processing complete!")

# Display the uploaded video
if video_file is not None:
    video_bytes = video_file.read()
    st.video(video_bytes)
