import cv2
import os
from os.path import isfile, join
import matplotlib.pyplot as plt

pathIn = r"img1" 
pathOut = "deneme.mp4"

files = [join(pathIn, f) for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

print(files[44])

img = cv2.imread(files[44])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)