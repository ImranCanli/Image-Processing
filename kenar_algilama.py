import cv2
import matplotlib.pyplot as plt
import numpy as np

#resmi ice aktarma
img = cv2.imread("image_processing.jpg", 0)
plt.figure()
plt.imshow(img, cmap= "gray")
plt.axis("off")
plt.title("Orijinal hali")
plt.show()


edges = cv2.Canny(image = img, threshold1=0, threshold2=255)
plt.figure()
plt.imshow(edges, cmap= "gray")
plt.axis("off")
plt.title("Kenarlar")
plt.show()


med_val = np.median(img)
print(med_val)

low = int(max(0, (1-0.33)*med_val))
high = int(min(255, (1+0.33)*med_val))
print(low)
print(high)


edgesNew = cv2.Canny(image = img, threshold1=150, threshold2=220)
plt.figure()
plt.imshow(edgesNew, cmap= "gray")
plt.axis("off")
plt.title("Kenarlar (duzenlenmis)")
plt.show()

blurred_img = cv2.blur(img, ksize = (3,3)) 
plt.figure()
plt.imshow(blurred_img, cmap= "gray")
plt.axis("off")
plt.title("Bulaniklastirilmis")
plt.show()

med_val_blurred = np.median(blurred_img)
print(med_val_blurred)

low_blurred = int(max(0, (1-0.33)*med_val_blurred))
high_blurred = int(min(255, (1+0.33)*med_val_blurred))
print(low_blurred)
print(high_blurred)


edgesNewBlurred = cv2.Canny(image = blurred_img, threshold1=low_blurred, threshold2=high_blurred)
plt.figure()
plt.imshow(edgesNewBlurred, cmap= "gray")
plt.axis("off")
plt.title("Kenarlar (bulaniklastirilmis)")
plt.show()
