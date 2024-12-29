import cv2
import matplotlib.pyplot as plt

#resmi ice aktar

img1 = cv2.imread("ricky_dicky.jpg")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
plt.figure()
plt.imshow(img1, cmap = "gray")
plt.axis("off")
plt.show()

#esikleme

_, thresh_img = cv2.threshold(img1, thresh = 60, maxval = 255, type = cv2.THRESH_BINARY_INV)

plt.figure()
plt.imshow(thresh_img, cmap= "gray")
plt.axis("off")
plt.show()


#uyarlamalı esik degeri

thresh_img2 = cv2.adaptiveThreshold(img1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 8)
plt.figure()
plt.imshow(thresh_img2, cmap = "gray")
plt.axis("off")
plt.show()