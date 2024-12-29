import cv2
import matplotlib.pyplot as plt
import numpy as np

# Resmi ice aktar
coin = cv2.imread("coins.jpg")
#coin = cv2.cvtColor(coin, cv2.COLOR_BGR2RGB)
plt.figure()
plt.title("Original Resim")
plt.axis("off")
plt.imshow(coin, cmap = "gray")


# Median blur
coin_blur = cv2.medianBlur(coin, 7)
plt.figure()
plt.title("Bulanik Resim")
plt.axis("off")
plt.imshow(coin_blur, cmap = "gray")

# Graysclae
coin_gray = cv2.cvtColor(coin_blur, cv2.COLOR_RGB2GRAY)
plt.figure()
plt.title("Siyah-Beyaz ve Bulanik Resim")
plt.axis("off")
plt.imshow(coin_gray, cmap = "gray")

# Binary threshold
ret, coin_thresh = cv2.threshold(coin_gray, 130, 225, cv2.THRESH_BINARY)
plt.figure()
plt.title("Objeler arkaplandan ayrilmis resim")
plt.axis("off")
plt.imshow(coin_thresh, cmap = "gray")

# Kontur
contours, hierarchy = cv2.findContours(coin_thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    
    if hierarchy[0][i][3] == -1: 
        cv2.drawContours(coin, contours, i,(0,255,0), 10)
        
plt.figure()
plt.title("Konturlari Bulunmus Resim")
plt.axis("off")
plt.imshow(coin, cmap = "gray")



# WATERSHED

# Resmi ice aktar yeniden
coin = cv2.imread("coins_new.jpg")
#coin = cv2.cvtColor(coin, cv2.COLOR_BGR2RGB)
plt.figure()
plt.title("Original Resim Yenided")
plt.axis("off")
plt.imshow(coin, cmap = "gray")


# Median blur yeniden
coin_blur = cv2.medianBlur(coin, 3)
plt.figure()
plt.title("Bulanik Resim Yeniden")
plt.axis("off")
plt.imshow(coin_blur, cmap = "gray")

# Graysclae Yeniden
coin_gray = cv2.cvtColor(coin_blur, cv2.COLOR_RGB2GRAY)
plt.figure()
plt.title("Siyah-Beyaz ve Bulanik Resim")
plt.axis("off")
plt.imshow(coin_gray, cmap = "gray")

# Binary threshold Yeniden
ret, coin_thresh = cv2.threshold(coin_gray, 200, 240, cv2.THRESH_BINARY)
plt.figure()
plt.title("Threshold uygulanmis resim")
plt.axis("off")
plt.imshow(coin_thresh, cmap = "gray")

# Acılma
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(coin_thresh, cv2.MORPH_OPEN, kernel, iterations = 2)

# Nesneler arasi distance
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
plt.figure()
plt.title("Uzaklik gosterimi")
plt.axis("off")
plt.imshow(dist_transform, cmap = "gray")

# Resmi Kucult
ret, sure_foreground = cv2.threshold(dist_transform, 0.4*np.max(dist_transform), 255, 0)
plt.figure()
plt.title("Nesnelerin Bulunmasi icin kucultulmus resim")
plt.axis("off")
plt.imshow(sure_foreground, cmap = "gray")


