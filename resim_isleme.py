import cv2 

# içe aktarma

img = cv2.imread("ricky_dicky.jpg", 0)

# gorsellestirme

cv2.imshow("Ilk Resim", img)

k = cv2.waitKey(0) &0xFF

if k == 27:
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite("rick_gray.jpg", img) 
    cv2.destroyAllWindows()