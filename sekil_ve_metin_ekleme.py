import cv2
import numpy as np

# resim olustur
resim = np.zeros((512,512, 3), np.uint8) # siyah resim
print(resim.shape)

cv2.imshow("Siyah resim", resim)

#cizgi 

cv2.line(resim, (0,0), (512,512), (0, 255, 0), 3) #resim, baslangic noktasi, bitis noktasi, renk 
cv2.imshow("Yesil cizgili", resim)

#dikdötgen
#resim, baslangic, bitis, renk
cv2.rectangle(resim, (0,0), ( 256,256), (255,0,0), cv2.FILLED)
cv2.imshow("Dikdortgenli", resim)

#cember
# resim, merkez, yaricap, renk

cv2.circle(resim, (300,300), 45, (0,0,255), cv2.FILLED)
cv2.imshow("daireli", resim)

#metin koyma
# resim, baslangic noktasi, font, yazı kalinligi, renk
cv2.putText(resim, "Resssim", (350,350), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255))
cv2.imshow("Yazili", resim)