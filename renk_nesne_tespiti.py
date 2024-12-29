import cv2
import matplotlib.pyplot as plt
from collections import deque


# Nesne merkezini dpolayacak veritipi
buffer_size = 16
pts = deque(maxlen = buffer_size)

# Mavi renk aralıgı HSV

blueLower = (0, 100, 100)
blueUpper = (30, 255, 255)

#resmi ice aktarma
img = cv2.imread("pens.jpg")

if img is None:
    print("Goruntu yuklenemedi")
else:
    # Goruntuyu hsv formatına cevir
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Mavi renge gore maske olustur
    mask = cv2.inRange(hsv, blueLower, blueUpper)
    
    # Maskeyi temizle  
    mask = cv2.erode(mask, None, iterations= 2)
    mask = cv2.dilate(mask, None, iterations= 2)
    
    #Konturlari bul
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
center = None
if len(contours) > 0:
    
    # En buyuk konturu bul
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Baska bir kontur bulma yontemi
    #contours = [c for c in contours if cv2.contourArea(c) > 100]
    
    # Minimum cevreleyen cemberi bul
    ((x,y), radius) = cv2.minEnclosingCircle(largest_contour)
    
    # Moment hesapla
    M = cv2.moments(largest_contour)
    # Alttaki satırı 
    #center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    
    if M["m00"] > 0:
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    else:
        center = None
    
    # yeterince buyukse daire ciz
    if radius > 10:
        cv2.circle(img, (int(x), int(y)), int(radius), (0,255,255), 2)
        cv2.circle(img, center, 5, (0,0,255), -1)
    
# Resmi goster
plt.figure()
plt.title("Tespit Edilen Nesne")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

# Maskeyi goster
plt.figure()
plt.title("Maske")
plt.imshow(mask, cmap="gray")
plt.axis("off")
plt.show()
