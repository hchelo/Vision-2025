import cv2
import numpy as np

# Cargar la imagen base y convertir a escala de grises
img1 = cv2.imread('fanta.jpg')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray1 = cv2.equalizeHist(gray1)

# Crear detector ORB
orb = cv2.ORB_create(nfeatures=1500)

# Detectar keypoints y descriptores en la imagen base
kp1, des1 = orb.detectAndCompute(gray1, None)

# Cargar el video
cap = cv2.VideoCapture('fanta3.mp4')

# Matcher para descriptores binarios
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# Empieza desde el frame 100
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
count_frame = 10

while True:
    ret, frame = cap.read()
    if not ret:
        break

    count_frame += 1
    if count_frame % 2 != 0:
        continue

    # Convertir frame a escala de grises
    gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.equalizeHist(gray2)

    # Detectar keypoints y descriptores en el frame
    kp2, des2 = orb.detectAndCompute(gray2, None)
    if des2 is None or len(des2) < 2:
        continue

    # Calcular coincidencias KNN
    try:
        matches = bf.knnMatch(des1, des2, k=2)
    except:
        continue

    # Ratio test de Lowe
    good_matches = []
    for m, n in matches:
        if m.distance < 0.70 * n.distance:
            good_matches.append(m)

    # Dibujar los buenos matches si hay suficientes
    if len(good_matches) > 10:
        matched_frame = cv2.drawMatches(img1, kp1, frame, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow('ORB Matches con Video', matched_frame)
    else:
        cv2.imshow('ORB Matches con Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
