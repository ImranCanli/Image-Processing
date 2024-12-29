import cv2
import numpy as np

img = cv2.imread("ricky_dicky.jpg")
cv2.imshow("FirstVer", img)

width = 400
height = 500

pts1 = np.float32([[230,1],[1,472], [540,150], [338,617]])
pts2 = np.float32([[0,0], [0, height], [width, 0], [width, height]])

matrix = cv2.getPerspectiveTransform(pts1, pts2)
print(matrix)

imgOutput =  cv2.warpPerspective(img, matrix, (width, height))
cv2.imshow("Nihai Resim", imgOutput)