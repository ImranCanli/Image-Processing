import cv2
import time 

# video ismi 
video_name = "v_clip2.mp4"

# video'yu içeri aktar
cap = cv2.VideoCapture(video_name)

print("genislik: ", cap.get(3))
print("yukseklik", cap.get(4))


if cap.isOpened() == False:
    print("Hata")


while True:
    
    ret, frame = cap.read()
    
    if ret == True:
        time.sleep(0.01) # Bu kullaılmazsa video cok hızlı akar
        
        cv2.imshow("Video", frame)
    else:
        break
    
    if cv2.waitKey(1) & 0XFF == ord("q"):
        break
    
    
cap.release() #video yakalamayı bırakır
cv2.destroyAllWindows()