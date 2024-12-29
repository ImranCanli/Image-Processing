import cv2
import os

files = os.listdir()

img_path_list = []

for f in files:
    if f.endswith("yaya.jpg"):
        img_path_list.append(f)
        
print(img_path_list)


# HOG tanımlayıcısı
hog = cv2.HOGDescriptor()

# Tanimlayiciya svm eklenmesi
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

for img_path in img_path_list:
    print(img_path)
    
    image = cv2.imread(img_path)
    
    (rects, weights) = hog.detectMultiScale(image, padding = (8,8), scale = 1.05 )
    
    for (x, y, w, h) in rects:
        cv2.rectangle(image, (x,y), (x+w, y+h), (0, 0, 255), 2)
    
    
    cv2.imshow("Yaya", image)
    if cv2.waitKey(0) & 0xFF == ord("q"): continue