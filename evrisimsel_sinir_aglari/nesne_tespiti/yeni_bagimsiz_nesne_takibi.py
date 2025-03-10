import cv2
import torch
import numpy as np
from yolov5 import YOLOv5
from deep_sort_realtime.deepsort_tracker import DeepSort
import pandas as pd

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.8
model.iou = 0.5

tracker = DeepSort(max_age=30, n_init=5)

cap = cv2.VideoCapture(0)

pre_positions = {}

while cap.isOpened():
    
    ret, frame = cap.read()

    if not ret:
        print("kamera yanit vermedi")
        break
    
    
    frame = cv2.resize(frame, (640, 640))
    
    # Buradan asagisini deneme icin ekledim...
    img = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    #deneme icin eklenen yerin sonu.
    
    results = model(frame)
    
    # Burası neden hata verdi anlamadim
    #detections = results.xyxy[0].cpu().numpy()
    
    df = results.pandas().xyxy[0]
    
    formatted_detections = []
    
    for index, row in df.iterrows():
        x1 = row['xmin']
        y1 = row['ymin']
        x2 = row['xmax']
        y2 = row['ymax']
        conf = row['confidence']
        cls = int(row['class'])
        
        if conf > 0.8:
            formatted_detections.append([[x1, y1, x2, y2], conf, int(cls)])

    tracked_objects = tracker.update_tracks(formatted_detections, frame = frame)
    
    for tracked in tracked_objects:
        
        if not tracked.is_confirmed():
            continue
        
        x1, y1, x2, y2 = tracked.to_tlbr()
        track_id = tracked.track_id
        
        x_center = (x1 + x2)/2
        y_center = (y1 + y2)/2

        if track_id in pre_positions:
            prev_x, prev_y = pre_positions[track_id]
        
            dx = x_center - prev_x
            dy = y_center - prev_y
        
            if abs(dx) > abs(dy):
                direction = "sag" if dx > 0 else "sol"
            else:
                direction = "asagi" if dy > 0 else "yukari"
            
            cv2.putText(frame, f"{direction}", (int(x1), int(y1)-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        pre_positions[track_id] = (x_center, y_center)
        
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {int(track_id)}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
     
    cv2.imshow("Nesne takibi", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("z"):
        break

cap.release()
cv2.destroyAllWindows()
         
    