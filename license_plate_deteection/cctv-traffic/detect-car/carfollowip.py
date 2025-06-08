import os
import random
import cv2
import numpy as np
import requests
from ultralytics import YOLO
from functions.tracker import Tracker

ip_path = "rtsp://service:Tonkla2249!@192.168.1.103:554/video"
video_out_path = os.path.join('.', 'out.mp4')

cap = cv2.VideoCapture(ip_path, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 30) 
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"H264"))

if not cap.isOpened():
    print("Not connect IP camera")
    exit()

ret,frame = cap.read()
if not ret:
    print("Unable read first frame")
    cap.release()
    exit()
    
cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'),cap.get(cv2.CAP_PROP_FPS), (frame.shape[1], frame.shape[0]))

model = YOLO("/Users/sarawit/license_plate_deteection/model/yolo11n.pt").to("cuda")
tracker = Tracker()

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(100)]
detection_threshold = 0.5

line_points = np.array([[307, 1484], [2264, 1495]])
pt1 = tuple(line_points[0])
pt2 = tuple(line_points[1])
line_tolerance = 10

crop_folder = "/Users/sarawit/license_plate_deteection/cctv-traffic/CarCrops"
if not os.path.exists(crop_folder):
    os.makedirs(crop_folder)

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Unable to read frame from camera (Possible buffer issue or disconnection)")
        break

    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    results = model(frame)

    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = map(int, r[:6])
            if score > detection_threshold and class_id in [2, 3, 5, 7]:
                detections.append([x1, y1, x2, y2, score])

        tracker.update(frame, detections)

        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = map(int, bbox)
            track_id = track.track_id
            cv2.rectangle(frame, (x1, y1), (x2, y2), colors[track_id % len(colors)], 3)
            cv2.putText(frame, f'ID: {track_id}', (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[track_id % len(colors)], 2)

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]
            distance = abs(dx * (pt1[1] - cy) - (pt1[0] - cx) * dy) / np.sqrt(dx**2 + dy**2)

    cv2.imshow("IP Camera Stream", frame)
    cap_out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cap_out.release()
cv2.destroyAllWindows()
