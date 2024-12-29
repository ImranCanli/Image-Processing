import cv2
import matplotlib.pyplot as plt
import numpy as np

#resmi ice aktarma
img = cv2.imread("image_processing.jpg", 0)
plt.figure()
plt.title("Orijinal hali")
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.show()

contours, hierarchy =  cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

external_contours = np.zeros(img.shape)
internal_contours = np.zeros(img.shape)

for i  in range(len(contours)):
    #external 
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(external_contours, contours, i, (255,40,40), -1)
    else:
        #internal
        cv2.drawContours(internal_contours, contours, i, (255,40,40), -1)
        
plt.figure()
plt.title("Koseleri")
plt.imshow(internal_contours, cmap="gray")
plt.axis("off")
plt.show()