from yolov5 import YOLOv5
# from deep_sort_realtime import deep_sort
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np

model = YOLOv5('yolov5s.pt')
model.conf = 0.7
model.iou = 0.5
# deepSort = deep_sort()
tracker = DeepSort(max_age=50, n_init=5)


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
        
    if not ret:
        print("kamera yanit vermedi")
        break
    
    frame = cv2.resize(frame, (640, 640))
    
    # görüntüyü modelin performansını arttırmak için düzenleme kısmı
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (5,5), 0)
    # edges = cv2.Canny(blurred, 50, 150)
    # frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    
    # nesnenin tespit edilmesi
    results = model.predict(frame)
    #results.show()
    
    detections = results.xyxy[0].cpu().numpy()
    formatted_detections = []
    
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        
        if conf > 0.8: 
            formatted_detections.append([[x1, y1, x2, y2], conf, int(cls)])
    
    # nesnelerin takip edilmesi 
    tracked_objects = tracker.update_tracks(formatted_detections, frame=frame)
    
    for tracked in tracked_objects:
        
        if not tracked.is_confirmed():
            continue
        
        x1, y1, x2, y2 = tracked.to_tlbr()
        track_id = tracked.track_id
        
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {int(track_id)}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
       
    cv2.imshow('Kamera', frame) # Pencere adını değiştir
    
    if cv2.waitKey(1) & 0xFF == ord("z"):
        break
    
cap.release()
cv2.destroyAllWindows()
