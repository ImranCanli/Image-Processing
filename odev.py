import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("image_processing.jpg", 0)
img_colored = cv2.imread("image_processing.jpg")
img_colored = cv2.cvtColor(img_colored, cv2.COLOR_BGR2RGB)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure()
plt.axis("off")
plt.imshow(img)
plt.title("Orijinal")
plt.show()


width, height, dim = img.shape
print("resmin genisligi: ", width)
print("resmin uzunlugu: ", height)
print("resmin biseyleri: ", dim)

n_width = int(width*(4/5))
n_height = int(height*(4/5))

imgResized = cv2.resize(img, (n_width, n_height))
plt.figure()
plt.axis("off")
plt.title("yeniden boyutlandirilmis resim")
plt.imshow(imgResized)
plt.show()

new_width, new_height, new_dim = imgResized.shape

print("yeni resmin genisligi: ", new_width)
print("yeni resmin uzunlugu: ", new_height)

cv2.putText(imgResized, "Imrrrrrran", (350,350), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255))
plt.figure()
plt.axis("off")
plt.title("metin eklenmis resim")
plt.imshow(imgResized)
plt.show()

_, thresh_img = cv2.threshold(img, thresh = 50, maxval = 255, type = cv2.THRESH_BINARY)
plt.figure()
plt.imshow(thresh_img, cmap= "gray")
plt.title("Threshold uygulanmis resim")
plt.axis("off")
plt.show()

gaussianBlur = cv2.GaussianBlur(img, ksize = (9,9), sigmaX = 9)
plt.figure()
plt.imshow(gaussianBlur)
plt.title("Gaussion blur uygulanmis resim")
plt.axis("off")
plt.show()

laplacian = cv2.Laplacian(img, ddepth = cv2.CV_16S)
plt.figure()
plt.imshow(laplacian)
plt.title("Laplacian uygulanmis resim")
plt.axis("off")
plt.show()

img_hist = cv2.calcHist([img], channels = [0], mask = None ,histSize= [256] , ranges = [0, 256])
print(img_hist.shape)
plt.figure()
plt.plot(img_hist)

color = ("b", "g", "r")
plt.figure()
for i, c in enumerate(color):
    hist = cv2.calcHist([img_colored], channels = [i], mask = None ,histSize= [256] , ranges = [0, 256])
    plt.plot(hist, color = c)