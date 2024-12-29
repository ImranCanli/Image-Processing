import cv2

img = cv2.imread("ricky_dicky.jpg")
imgbw = cv2.imread("ricky_dicky.jpg", 0)
print("Resim Boyutu: ", img.shape)
cv2.imshow("Orijinal", imgbw)

#resize
imgResized = cv2.resize(img, (800,800))
print("Resized Image Shape", imgResized.shape)
cv2.imshow("Yeniden boyutlandirilmis fotograf: ", imgResized)


#kırp

imgCropped = img[:200, 0:300]
cv2.imshow("Kirpik resim: ", imgCropped)