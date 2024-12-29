import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("sudoku.jpg")
plt.figure()
plt.axis("off")
plt.imshow(img, cmap = "gray")
plt.title("Orijinal Resim")
plt.show()


#X gradyan
sobelx = cv2.Sobel(img, ddepth = cv2.CV_16S, dx = 1, dy = 0, ksize = 5)
plt.figure()
plt.axis("off")
plt.imshow(sobelx, cmap = "gray")
plt.title("Sobel X Resmi")
plt.show()


#Y gradyan
sobely = cv2.Sobel(img, ddepth = cv2.CV_16S, dx = 0, dy = 1, ksize = 5)
plt.figure()
plt.axis("off")
plt.imshow(sobely, cmap = "gray")
plt.title("Sobel Y Resmi")
plt.show()

#Lablacian gradyan
laplacian = cv2.Laplacian(img, ddepth = cv2.CV_16S)
plt.figure()
plt.axis("off")
plt.imshow(laplacian, cmap = "gray")
plt.title("Laplacian Resmi")
plt.show()
