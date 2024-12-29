import cv2
import matplotlib.pyplot as plt
from collections import deque
import numpy as np

# Template macthing

img = cv2.imread("image_processing.jpg", 0)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(img.shape)
plt.figure()
plt.title("Ana Resim")
plt.imshow(img, cmap = "gray")
plt.axis("off")
plt.show()


template = cv2.imread("image_processing_little.jpg", 0)
#template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
print(img.shape)
plt.figure()
plt.title("Kırpılmıs Resim")
plt.imshow(template, cmap = "gray")
plt.axis("off")
plt.show()

h, w = template.shape

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']


for meth in methods:
    
    method = eval(meth)
    
    res = cv2.matchTemplate(img, template, method)
    print(res.shape)
    
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        
        top_left = min_loc
        
    else:
        top_left = max_loc
        
    bottom_right = (top_left[0] + w, top_left[1] + h)
    
    cv2.rectangle(img, top_left, bottom_right, 255, 2)
    
    plt.figure()
    plt.subplot(121)
    plt.title("Eslesen Sonuc")
    plt.imshow(res, cmap = "gray")
    plt.axis("off")

    plt.subplot(122)
    plt.title("Tespit Edilen Sonuc")
    plt.imshow(img, cmap = "gray")
    plt.axis("off")
    
    plt.suptitle(meth)

    plt.show()
