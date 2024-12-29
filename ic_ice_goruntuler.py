import cv2
import matplotlib.pyplot as plt

#karıstırma

img1 = cv2.imread("ricky_dicky.jpg")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread("ricky_dicky_2.jpg")
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

plt.figure()
plt.imshow(img1)

plt.figure()
plt.imshow(img2)

print(img1.shape)
print(img2.shape)


#karıstırılmıs resim = alpha*img1 + beta*img2

blended = cv2.addWeighted(src1 = img1, alpha = 0.5, src2 = img2, beta = 0.5, gamma = 0)
plt.figure()
plt.imshow(blended)