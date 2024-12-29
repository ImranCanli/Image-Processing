import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("futbol_takimi.jpg")
img_clean_for_body_recognizing = cv2.imread("futbol_takimi.jpg")
img_clean_for_body_recognizing = cv2.cvtColor(img_clean_for_body_recognizing, cv2.COLOR_BGR2RGB)

plt.figure()
plt.axis("off")
plt.title("Original Version")
plt.imshow(img, cmap="gray")
plt.show()


# Kenar algilama

edges = cv2.Canny(img, threshold1=0, threshold2=255)
plt.figure()
plt.axis("off")
plt.title("Canny Version")
plt.imshow(edges, cmap="gray")
plt.show()

med_val = np.median(img)
print(med_val)

low = int(max(0, (1-0.33)*med_val))
high = int(min(255, (1+0.33)*med_val))
print(low)
print(high)

edges_after = cv2.Canny(img, threshold1=150, threshold2=220)
plt.figure()
plt.imshow(edges_after, cmap="gray")
plt.axis("off")
plt.title("Kenarlar (duzenlenmis)")
plt.show()

# Yuz tanima icin gerekli haarcascade'in iceri aktarilmasi
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Tespit edilen yuzlerin etrafina dikgortgen cizimi
face_rect = face_cascade.detectMultiScale(img, minNeighbors=4)

for (x, y, w, h) in face_rect:
    cv2.rectangle(img, (x,y), (x+w, y+h), (57,255,20), 10)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure()
plt.axis("off")
plt.title("Yuzler tepit edildi roketler atesleniyor...")
plt.imshow(img)
plt.show()


# HOG ice aktarimi
hog = cv2.HOGDescriptor()

hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

(rects, weights) = hog.detectMultiScale(img_clean_for_body_recognizing, padding = (3,3), scale = 1.04)

for (x, y, w, h) in rects:
    cv2.rectangle(img_clean_for_body_recognizing, (x,y), (x+w, y+h), (0, 0, 255), 2)


plt.figure()
plt.axis("off")
plt.title("Insanlarin tespit edilmis hali")
plt.imshow(img_clean_for_body_recognizing)
plt.show()
