import cv2
import matplotlib.pyplot as plt
import numpy as np

import warnings

warnings.filterwarnings("ignore")

#blurring detayı azaltıp, gürültüyü engeller
img = cv2.imread("ricky_dicky.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure()
plt.axis("off")
plt.title("orijinal")
plt.imshow(img)
#plt.show()

# ortalama bulanıklastirma

dst2 = cv2.blur(img, ksize=(5,5))
plt.figure()
plt.axis("off")
plt.title("ortalama blur")
plt.imshow(dst2)

# gaussian blur

dst3 = cv2.GaussianBlur(img, ksize = (3, 3), sigmaX = 7)
plt.figure()
plt.axis("off")
plt.title("Gaussian blur")
plt.imshow(dst3)

# medyan blur

dst4 = cv2.medianBlur(img, ksize = 3)
plt.figure()
plt.axis("off")
plt.title("Median blur")
plt.imshow(dst4)


def gaussianNoise(image):
    row, col, ch = image.shape
    mean = 0
    var = 0.05
    sigma = var**0.5
    
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    
    return noisy

#ice aktar ve normalize et

img = cv2.imread("ricky_dicky.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255
plt.figure()
plt.axis("off")
plt.title("Original")
plt.imshow(img)


gaussianNoisyImage = gaussianNoise(img)
plt.figure()
plt.axis("off")
plt.title("Gaussian Noised")
plt.imshow(gaussianNoisyImage)


dst5 = cv2.GaussianBlur(img, ksize = (3, 3), sigmaX = 7)
plt.figure()
plt.axis("off")
plt.title("with Gaussian blur")
plt.imshow(dst5)


# tuz biber cart curt

def saltPepperNoise(image, s_vs_p=0.3, amount=0.004):
    row, col, ch = image.shape
    noisy = np.copy(image)
    
    # Salt (beyaz noktalar)
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i, int(num_salt)) for i in image.shape]
    noisy[tuple(coords)] = 1  # NumPy'nin tuple ile indekslemeyi desteklediği format
    
    # Pepper (siyah noktalar)
    num_pepper = np.ceil(amount * image.size * (1 - s_vs_p))
    coords = [np.random.randint(0, i, int(num_pepper)) for i in image.shape]
    noisy[tuple(coords)] = 0
    
    return noisy

# Görüntüyü yükle ve normalize et
img = cv2.imread("ricky_dicky.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0  # Normalize edilmiş görüntü

# Salt and Pepper gürültüsü ekle
spImage = saltPepperNoise(img)

# Görüntüyü göster
plt.figure()
plt.axis("off")
plt.title("Sp Image")
plt.imshow(spImage)
plt.show()

