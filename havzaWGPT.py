import cv2
import matplotlib.pyplot as plt
import numpy as np

# 1. RESMİ OKUMA
coin = cv2.imread("coins_new.jpg")
plt.figure()
plt.title("Orijinal Resim")
plt.axis("off")
plt.imshow(cv2.cvtColor(coin, cv2.COLOR_BGR2RGB))

# 2. BULANIKLAŞTIRMA (Gürültüyü azaltmak için)
coin_blur = cv2.medianBlur(coin, 5)

# 3. GRAYSCALE DÖNÜŞÜMÜ
coin_gray = cv2.cvtColor(coin_blur, cv2.COLOR_BGR2GRAY)

# 4. HISTOGRAM EŞİTLEME (Kontrastı artırmak için)
coin_gray_eq = cv2.equalizeHist(coin_gray)
plt.figure()
plt.title("Histogram Eşitleme Sonrası")
plt.axis("off")
plt.imshow(coin_gray_eq, cmap="gray")

# 5. ADAPTİF THRESHOLDING
coin_thresh = cv2.adaptiveThreshold(
    coin_gray_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 2
)
plt.figure()
plt.title("Adaptif Thresholding Sonrası")
plt.axis("off")
plt.imshow(coin_thresh, cmap="gray")

# 6. MORFOLOJİK AÇILMA (Küçük gürültüleri temizlemek için)
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(coin_thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# 7. ARKA PLAN VE ÖN PLAN TESPİTİ
# Belirgin arka plan
sure_bg = cv2.dilate(opening, kernel, iterations=3)
plt.figure()
plt.title("Belirgin Arka Plan")
plt.axis("off")
plt.imshow(sure_bg, cmap="gray")

# Belirgin ön plan (Distance Transform)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
plt.figure()
plt.title("Belirgin Ön Plan")
plt.axis("off")
plt.imshow(sure_fg, cmap="gray")

# Belirsiz alan (arka plan - ön plan farkı)
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)
plt.figure()
plt.title("Bilinmeyen Bölge")
plt.axis("off")
plt.imshow(unknown, cmap="gray")

# 8. HSV RENK MASKELEME (Gümüş renkleri maskelemek için)
coin_hsv = cv2.cvtColor(coin_blur, cv2.COLOR_BGR2HSV)
lower_silver = np.array([0, 0, 180])  # Daha düşük doygunluk ve yüksek değer
upper_silver = np.array([180, 30, 255])
silver_mask = cv2.inRange(coin_hsv, lower_silver, upper_silver)

plt.figure()
plt.title("Gümüş Renk Maskeleme")
plt.axis("off")
plt.imshow(silver_mask, cmap="gray")

# 9. MARKER TANIMLAMA (Watershed için)
ret, markers = cv2.connectedComponents(sure_fg)

# Markerları güncelleme: Bilinmeyen bölgeyi işaretle (-1)
markers = markers + 1
markers[unknown == 255] = 0

# 10. WATERSHED ALGORİTMASI
coin_watershed = coin.copy()
cv2.watershed(coin_watershed, markers)

# Kenarları belirlemek için
coin_watershed[markers == -1] = [255, 0, 0]

# 11. SONUÇLARI GÖSTERME
plt.figure()
plt.title("Watershed Sonuç")
plt.axis("off")
plt.imshow(cv2.cvtColor(coin_watershed, cv2.COLOR_BGR2RGB))

plt.show()
