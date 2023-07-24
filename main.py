import cv2
from speed_calc import *
import numpy as np
from object_detection import ObjectDetection
import math

end = 0

# Create Tracker Object
tracker = EuclideanDistTracker()

# Initialize Object Detection
od = ObjectDetection()

# Initialize count
count = 0
center_points_prev_frame = []

tracking_objects = {}
track_id = 0

# Taking input
distance = int(input("Enter the distance:"))# 1332
pixel = int(input("Enter the video quality:"))# 1080

# Calculating distance pixel ratio
ratio = distance / pixel

cap = cv2.VideoCapture("Resources/traffic4.mp4") #Enter the file path of video like shown
f = 25
w = int(1000 / (f - 1))

# Create a window to display the video frames
cv2.namedWindow('frame')

# Define callback function to handle mouse events
def select_rectangle(event, x, y, flags, param):
    global rect_start, rect_end, selecting

    if event == cv2.EVENT_LBUTTONDOWN:
        rect_start = (x, y)
        selecting = True

    elif event == cv2.EVENT_LBUTTONUP:
        rect_end = (x, y)
        selecting = False

# Register the mouse callback function
cv2.setMouseCallback('frame', select_rectangle)

# Initialize the selecting flag
selecting = False

# Initial mouse coordinates for ROI
a = 0
b = 0
c = 1920
d = 1080
ht = d
wt = c

while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 3 != 0:
        continue

    # Region of Interest (ROI)
    roi = frame[b:d, a:c]

    # Point current frame
    center_points_cur_frame = []
    detections = []

    # Detect objects on frame
    (class_ids, scores, boxes) = od.detect(roi)
    for box in boxes:
        (x, y, w, h) = box
        cx = int((x + x + w) / 2)
        cy = int((y + y + h) / 2)
        center_points_cur_frame.append((cx, cy))
        detections.append([x, y, w, h])

        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Only at the beginning we compare previous and current frame
    if count <= 2:
        for pt in center_points_cur_frame:
            for pt2 in center_points_prev_frame:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                if distance < 20:
                    tracking_objects[track_id] = pt
                    track_id += 1
    else:
        tracking_objects_copy = tracking_objects.copy()
        center_points_cur_frame_copy = center_points_cur_frame.copy()

        for object_id, pt2 in tracking_objects_copy.items():
            object_exists = False
            for pt in center_points_cur_frame_copy:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                # Update IDs position
                if distance < 20:
                    tracking_objects[object_id] = pt
                    object_exists = True
                    if pt in center_points_cur_frame:
                        center_points_cur_frame.remove(pt)
                    continue

            # Remove IDs lost
            if not object_exists:
                tracking_objects.pop(object_id)

        # Add new IDs found
        for pt in center_points_cur_frame:
            tracking_objects[track_id] = pt
            track_id += 1

    # Speed calculation distance
    dist = ratio * (0.5 * ht)

    # Object Tracking
    boxes_ids = tracker.update(detections, ht)
    for box_id in boxes_ids:
        x, y, w, h, obj_id = box_id

        if tracker.get_speed(obj_id, dist) < tracker.get_speed_limit():
            cv2.putText(roi,str(tracker.get_speed(obj_id, dist)),(x,y-15), cv2.FONT_HERSHEY_PLAIN,1,(255,255,0),2)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
            
        else:
            cv2.putText(roi,str(tracker.get_speed(obj_id, dist)),(x, y-15),cv2.FONT_HERSHEY_PLAIN, 1,(0, 0, 255),2)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 165, 255), 3)

        s = tracker.get_speed(obj_id, dist)
        if tracker.f[obj_id] == 1 and s != 0:
            tracker.capture(roi, x, y, h, w, s, obj_id)

    # Draw the selected rectangle
    if not selecting and 'rect_start' in globals() and 'rect_end' in globals():
        cv2.rectangle(frame, rect_start, rect_end, (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        cv2.imshow('ROI', roi)
        a, b = rect_start
        c, d = rect_end
        ht = d - b
        wt = c - a

    # Draw lines
    cv2.line(roi, (0, int(0.625 * ht)), (int(wt), int(0.625 * ht)), (0, 0, 255), 2)
    cv2.line(roi, (0, int(0.75 * ht)), (int(wt), int(0.75 * ht)), (0, 0, 255), 2)
    cv2.line(roi, (0, int(0.375 * ht)), (int(wt), int(0.375 * ht)), (0, 0, 255), 2)
    cv2.line(roi, (0, int(0.25 * ht)), (int(wt), int(0.25 * ht)), (0, 0, 255), 2)

    # Display frames
    cv2.imshow("frame", frame)

    # Make a copy of the points
    center_points_prev_frame = center_points_cur_frame.copy()

    key = cv2.waitKey(w - 10)
    if key == 27:
        tracker.end()
        end = 1
        break

if end != 1:
    tracker.end()

cap.release()
cv2.destroyAllWindows()
