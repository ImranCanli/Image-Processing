import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("ricky_dicky.jpg", 0)
plt.figure()
plt.axis("off")
plt.imshow(img, cmap="gray")
plt.title("Orijinal Resim")

#erozyon

kernel = np.ones((5,5), dtype= np.uint8)
result = cv2.erode(img, kernel, iterations= 6)
plt.figure()
plt.axis("off")
plt.imshow(result, cmap="gray")
plt.title("Erozyonlu Resim")

# genisleme dilation

result2 = cv2.dilate(img, kernel, iterations= 6) 
plt.figure()
plt.axis("off")
plt.imshow(result2, cmap="gray")
plt.title("Tekrar Genisletilmis Resim")

# white noise

whiteNoise = np.random.randint(0, 2, size = img.shape[:2] )
whiteNoise = whiteNoise*255
plt.figure()
plt.imshow(whiteNoise, cmap="gray")
plt.axis("off")
plt.title("Beyaz gurultu")


noiseImage = whiteNoise + img
plt.figure()
plt.imshow(noiseImage, cmap="gray")
plt.axis("off")
plt.title("Beyaz gurultulu Resim")


# acilma

opening = cv2.morphologyEx(noiseImage.astype(np.float32), cv2.MORPH_OPEN, kernel)
plt.figure()
plt.imshow(opening, cmap="gray")
plt.axis("off")
plt.title("Beyaz gurultulu Acilmali Resim")


# siyah noise

blackNoise = np.random.randint(0, 2, size = img.shape[:2] )
blackNoise = blackNoise*(-255)
plt.figure()
plt.imshow(blackNoise, cmap="gray")
plt.axis("off")
plt.title("Kara gurultu")

blackNoiseImage = blackNoise + img

blackNoiseImage[blackNoiseImage < -245] = 0
plt.figure()
plt.imshow(blackNoiseImage, cmap="gray")
plt.axis("off")
plt.title("Siyah gurultulu Resim")

#gappatmaaaa

closing = cv2.morphologyEx(blackNoiseImage.astype(np.float32), cv2.MORPH_CLOSE, kernel)
plt.figure()
plt.imshow(closing, cmap="gray")
plt.axis("off")
plt.title("Siyah Gurultulu Kapatmali Resim")

# morfolojik gradient

gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
plt.figure()
plt.imshow(gradient, cmap="gray")
plt.axis("off")
plt.title("Gradientli Resim")
