import cv2
import matplotlib.pyplot as plt
import numpy as np

def non_max_supression(boxes, probs = None, overlapThresh = 0.3):
    
    if len(boxes) == 0:
        return[]
    
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
        
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    
    # alan hesaplanmasi
    
    area = (x2-x1+1)*(y2-y1+1)
    
    idxs = y2
    
    # probability
    
    if probs is not None:
        idxs = probs
        
    idxs = np.argsort(idxs)
    
    
    pick = []
    
    while len(idxs) > 0:
        last = len(idxs) - 1  
        i = idxs[last]
        pick.append(i)
        
        # en buyuk x ve y degerlerini bul
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        # genislik ve yukseklik bul
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
    
        # overlap bul
        overlap = (w*h)/area[idxs[:last]]
        
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
        
    return boxes[pick].astype("int")