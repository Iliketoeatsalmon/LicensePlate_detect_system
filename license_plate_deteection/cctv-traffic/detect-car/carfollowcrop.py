import os
import random
import cv2
import numpy as np
from ultralytics import YOLO
from functions.tracker import Tracker

video_path = "video/kmitl.mp4"
video_out_path = os.path.join('.', 'out.mp4')

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'),cap.get(cv2.CAP_PROP_FPS), (frame.shape[1], frame.shape[0]))

model = YOLO("model/yolo11n.pt")
tracker = Tracker()

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(10)]
detection_threshold = 0.5

line_points = np.array([[307, 1484], [2264, 1495]])
pt1 = tuple(line_points[0])
pt2 = tuple(line_points[1])
line_tolerance = 10

crop_folder = "CarCrops"
if not os.path.exists(crop_folder):
    os.makedirs(crop_folder)

#for storage vehicle
cropped_ids = set()

while ret:
    # วาดเส้นตรงลงบนเฟรม
    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    results = model(frame)

    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if score > detection_threshold and int(class_id) in [2, 3, 5, 7]:
                detections.append([x1, y1, x2, y2, score])

        tracker.update(frame, detections)

        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = map(int, bbox)
            track_id = track.track_id
            cv2.rectangle(frame, (x1, y1), (x2, y2), colors[track_id % len(colors)], 3)
            cv2.putText(frame, f'ID: {track_id}', (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[track_id % len(colors)], 2)

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            dx = pt2[0] - pt1[0]
            dy = pt2[1] - pt1[1]
            distance = abs(dx * (pt1[1] - cy) - (pt1[0] - cx) * dy) / np.sqrt(dx**2 + dy**2)

            if distance <= line_tolerance:
                if track_id not in cropped_ids:
                    crop_img = frame[y1:y2, x1:x2]
                    filename = os.path.join(crop_folder, f"car_{track_id}.jpg")
                    cv2.imwrite(filename, crop_img)
                    cropped_ids.add(track_id)
    cap_out.write(frame)
    ret, frame = cap.read()
cap.release()
cap_out.release()
cv2.destroyAllWindows()