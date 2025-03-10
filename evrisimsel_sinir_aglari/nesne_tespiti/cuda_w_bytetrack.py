import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import pandas as pd
# from yolox.tracker.byte_tracker import BYTETracker


# YOLOv8 modelini yükle
model = YOLO('yolov8s')  # veya yolov8n, yolov8m, yolov8l gibi diğer boyutlar

model.conf = 0.7  # Güven eşiği
model.iou = 0.5  # IoU eşiği

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # CUDA varsa 'cuda', yoksa 'cpu' kullanır.
model = model.to(device)

# tracker = BYTETracker

tracker.args = {
    "track_thresh": 0.5,
    "match_thresh": 0.8,
    "track_buffer": 30}


cap = cv2.VideoCapture(0)

pre_positions = {}

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Kamera yanıt vermedi")
        break

    frame = cv2.resize(frame, (640, 640))
    print(frame.shape)
    
    # ilk bastaki frame donusturme 
    # frame_cuda = torch.from_numpy(frame).to(device)
    
    # ikinci frame donusturme
    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().to(device)
    
    # YOLOv8 ile nesne tespiti
    results = model(frame_tensor)

    detections = []
    
    for *xyxy, conf, cls in results[0].boxes.data:
        detections.append([*xyxy, conf])
    
    byte_detections = np.array(detections)
    
    # df = pd.DataFrame(detections, columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class'])

    # formatted_detections = []

    # for index, row in df.iterrows():
    #     x1 = row['xmin']
    #     y1 = row['ymin']
    #     x2 = row['xmax']
    #     y2 = row['ymax']
    #     conf = row['confidence']
    #     cls = int(row['class'])

    #     if conf > 0.7:
    #         formatted_detections.append([[x1, y1, x2, y2], conf])

    tracked_objects = tracker.update(byte_detections, frame.shape)

    for tracked in tracked_objects:

        # if not tracked.is_confirmed():
        #     continue

        x1, y1, x2, y2, track_id = tracked[:5]
        track_id = tracked.track_id

        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2

        if track_id in pre_positions:
            prev_x, prev_y = pre_positions[track_id]

            dx = x_center - prev_x
            dy = y_center - prev_y

            direction = "sag" if dx > 0 else "sol" if abs(dx) > abs(dy) else "asagi" if dy > 0 else "yukari"
            cv2.putText(frame, f"{direction}", (int(x1), int(y1) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)


        pre_positions[track_id] = (x_center, y_center)

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {int(track_id)}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    cv2.imshow("Nesne takibi", frame)

    if cv2.waitKey(1) & 0xFF == ord("z"):
        break

cap.release()
cv2.destroyAllWindows()