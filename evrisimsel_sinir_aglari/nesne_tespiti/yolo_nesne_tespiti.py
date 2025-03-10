import cv2
import numpy as np
from yolo_model import YOLO

yolo = YOLO(0.6, 0.5)
file = "coco_classes.txt"

with open(file) as f:
    
    class_name = f.readlines()