import cv2
import numpy as np

# Kamerayı başlat
cap = cv2.VideoCapture(0)

# Çizim için boş bir siyah tuval oluştur (Kamera görüntüsüyle aynı boyutta)
canvas = None

# Mavi renk için HSV sınırları (Bu değerleri elindeki nesneye göre güncelleyebiliriz)
lower_blue = np.array([100, 150, 50])
upper_blue = np.array([140, 255, 255])

# Önceki koordinatları tutmak için (Çizgi çekmek için)
prev_x, prev_y = 0, 0

while True:
    success, frame = cap.read()
    if not success: break
    
    frame = cv2.flip(frame, 1) # Aynalama
    if canvas is None:
        canvas = np.zeros_like(frame)

    # Görüntüyü HSV formatına çevir (Renk algılama için daha iyidir)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Sadece mavi renkleri maskele
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.erode(mask, None, iterations=2) # Gürültüyü temizle
    mask = cv2.dilate(mask, None, iterations=2)

    # Maskelenmiş bölgedeki en büyük objeyi bul
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # En büyük konturu bul (Senin elindeki nesne odur)
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        
        if radius > 10: # Eğer nesne yeterince büyükse
            curr_x, curr_y = int(x), int(y)
            
            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = curr_x, curr_y
            
            # Tuval üzerine çizgi çiz
            cv2.line(canvas, (prev_x, prev_y), (curr_x, curr_y), (255, 0, 0), 5)
            prev_x, prev_y = curr_x, curr_y
    else:
        prev_x, prev_y = 0, 0

    # Kameradan gelen görüntü ile çizim yaptığımız tuvali birleştir
    combined = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)

    cv2.imshow("Mavi Bir Nesneyle Cizim Yap!", combined)
    
    # 'c' tuşuna basınca ekranı temizle
    if cv2.waitKey(1) & 0xFF == ord('c'):
        canvas = np.zeros_like(frame)
    
    # 'q' tuşuna basınca çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()