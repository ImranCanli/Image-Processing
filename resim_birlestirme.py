import cv2
import numpy as np

img = cv2.imread("ricky_dicky.jpg")
cv2.imshow("Orijinal", img)

resizedHor = cv2.resize(img, (250, 300))
hor = np.hstack((resizedHor, resizedHor))
cv2.imshow("Horizontal", hor)

dikey = np.vstack((resizedHor, resizedHor))
cv2.imshow("Dikey",dikey)