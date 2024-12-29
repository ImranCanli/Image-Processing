import cv2
import matplotlib.pyplot as plt
#import numpy as np

# 1. RESMİ OKUMA
human = cv2.imread("human_face.jpg", 0)
plt.figure()
plt.title("Orijinal Resim")
plt.axis("off")
plt.imshow(human, cmap = "gray")

# Siniflandirici
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

face_rectangle =  face_cascade.detectMultiScale(human)

for (x, y, w, h) in face_rectangle:
    cv2.rectangle(human, (x,y), (x+w, y+h), (80,120,40), 10)

plt.figure()
plt.title("Yuz Tespiti")
plt.axis("off")
plt.imshow(human, cmap = "gray")


# Birden fazla yüz tespiti


# Birden fazla yuz tespiti icin resim
some_team = cv2.imread("futbol_takimi.jpg", 0)
plt.figure()
plt.title("Futbol takimi resim")
plt.axis("off")
plt.imshow(some_team, cmap = "gray")

# Siniflandirici
face_rectangle =  face_cascade.detectMultiScale(some_team, minNeighbors = 4)

for (x, y, w, h) in face_rectangle:
    cv2.rectangle(some_team, (x,y), (x+w, y+h), (20,240,20), 10)

plt.figure()
plt.title("Yuz Tespiti")
plt.axis("off")
plt.imshow(some_team, cmap = "gray")





# Kamera ile yuz tespiti

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if ret:
        face_rectangle =  face_cascade.detectMultiScale(frame, minNeighbors = 4)
        
        for (x, y, w, h) in face_rectangle:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,255,255), 10)
        
        cv2.imshow("face detect", frame)
        
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()


# Video ile yüz tanima

cap = cv2.VideoCapture("for_face_recognition.mp4")
if not cap.isOpened():
    print("Video Bulunamadi")
    exit()

# Videonun kare kare yüz tespiti

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Video bitti ve ya bir hata oluştu")
        break
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Yuz tespitinin yapilmasi
    face_rectangle = face_cascade.detectMultiScale(gray_frame, minNeighbors = 4)
    
    # Tespit edilen yuzlerin etrafına dikdortgen cizilmesi
    for (x, y, w, h) in face_rectangle:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow("Yuz tespiti Video", frame)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()