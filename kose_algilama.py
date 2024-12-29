import cv2
import matplotlib.pyplot as plt
import numpy as np

#resmi ice aktarma
img = cv2.imread("sudoku.jpg", 0)
img = np.float32(img)
print(img.shape)
plt.figure()
plt.title("Orijinal hali")
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.show()

#harris corner detection
dst = cv2.cornerHarris(img, blockSize = 2, ksize = 3, k = 0.04)
plt.figure()
plt.title("Koseleri")
plt.imshow(dst, cmap="gray")
plt.axis("off")
plt.show()

dst = cv2.dilate(dst, None)
img[dst > 0.2*dst.max()] = 1
plt.figure()
plt.title("Koseleri ama belirginleştirilmiş")
plt.imshow(dst, cmap="gray")
plt.axis("off")
plt.show()

#shi tomasi detection
img = cv2.imread("sudoku.jpg", 0)
img = np.float32(img)
corners = cv2.goodFeaturesToTrack(img, 100, 0.01, 10)
corners = np.int64(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(img, (x,y), 3, (20,20,255), cv2.FILLED)

plt.figure()
plt.title("Yeni algoritme ile koseler")
plt.imshow(img)
plt.axis("off")
plt.show()