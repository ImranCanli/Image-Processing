import cv2
import os

"""files = os.listdir()
#print(files)

img_path_list = []
for f in files:
    if f.endswith(".jpg"):
        img_path_list.append(f)

print(img_path_list)"""

spesific_files = {"cats.jpg", "cat1.jpg", "cat2.jpg"}

files = os.listdir()

img_path_list = [f for f in files if f in spesific_files]

print(img_path_list)

for J in img_path_list:
    print(J)
    image = cv2.imread(J)
    
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    detector = cv2.CascadeClassifier("haarcascade_frontalcatface.xml")
    rects = detector.detectMultiScale(gray, scaleFactor = 1.045, minNeighbors = 3)
    
    for (i, (x, y, w, h)) in enumerate(rects):
        cv2.rectangle(image, (x,y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, "Kedi{}".format(i+ 1), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
     
    
    cv2.imshow(J, image)
    if cv2.waitKey(0) & 0xFF == ord("q"): continue

    