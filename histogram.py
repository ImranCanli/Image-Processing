import cv2
import matplotlib.pyplot as plt

img = cv2.imread("image_processing.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure()
plt.axis("off")
plt.imshow(img, cmap = "gray")
plt.title("Orijinal Resim")
plt.show()

print(img.shape)

img_hist = cv2.calcHist([img], channels = [0], mask = None ,histSize= [256] , ranges = [0, 256])
print(img_hist.shape)
plt.figure()
plt.plot(img_hist)

color = ("b", "g", "r")
plt.figure()
for i, c in enumerate(color):
    hist = cv2.calcHist([img], channels = [i], mask = None ,histSize= [256] , ranges = [0, 256])
    plt.plot(hist, color = c)