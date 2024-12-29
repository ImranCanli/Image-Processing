import cv2
import matplotlib.pyplot as plt

# Ana goruntuyu ice aktar
chos = cv2.imread("chocolates.jpg", 0)

plt.figure()
plt.axis("off")
plt.title("Original Tum Resim")
plt.imshow(chos, cmap = "gray")

# Aranacak olan gorsel
cho = cv2.imread("Kinder-Bueno.jpg", 0)

plt.figure()
plt.title("Arancak Olan Gorsel")
plt.axis("off")
plt.imshow(cho, cmap = "gray")


# Orb Tanımlayıcısı
# Kose-Kenar gibi nesneye ait ozellikler
orb = cv2.ORB_create()

# Anahtar nokta tespiti
kp1, des1 = orb.detectAndCompute(cho,  None)
kp2, des2 = orb.detectAndCompute(chos, None)

# Brute Force matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

# Noktalari eslestir
matches = bf.match(des1, des2)

# Mesafeye gore sırala
matches = sorted(matches, key = lambda x: x.distance )


# Eslesen resimleri gorsellestirme
img_smthng = cv2.drawMatches(cho, kp1, chos, kp2, matches[:20], None, flags = 2)
plt.figure()
plt.title("Karsilasma Sonuclari")
plt.axis("off")
plt.imshow(img_smthng, cmap = "gray")


# Sift
sift = cv2.SIFT_create()

# bf
bf = cv2.BFMatcher()

# Sift ile anahtar nokta tespiti
kp1, des1 = sift.detectAndCompute(cho,  None)
kp2, des2 = sift.detectAndCompute(chos, None)

matches = bf.knnMatch(des1, des2, k = 2)

guzel_eslesme = []

for match1, match2 in matches:
    
    if match1.distance > 0.75*match2.distance:
        guzel_eslesme.append([match1])
        
    
plt.figure()
sift_matches = cv2.drawMatchesKnn(cho, kp1, chos, kp2, guzel_eslesme, None, flags = 2)
plt.title("Eslesme Yeni Yontem") 
plt.axis("off")
plt.imshow(sift_matches, cmap = "gray") 
     