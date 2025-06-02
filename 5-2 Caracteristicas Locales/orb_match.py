import cv2

# Cargar imágenes
img1 = cv2.imread('fanta.jpg')
img2 = cv2.imread('fantaB.png')

# Convertir a escala de grises
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Ecualización del histograma (opcional, mejora contraste)
gray1 = cv2.equalizeHist(gray1)
gray2 = cv2.equalizeHist(gray2)

# Crear objeto ORB
orb = cv2.ORB_create(nfeatures=3500)

# Detectar características y descripciones
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

# Crear BFMatcher con Hamming (para ORB)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# Buscar los 2 mejores matches con KNN
matches = bf.knnMatch(des1, des2, k=2)

# Ratio test de Lowe
good_matches = []
for m, n in matches:
    if m.distance < 0.65 * n.distance:
        good_matches.append(m)

# Dibujar los buenos matches
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Mostrar resultado
cv2.imshow('ORB: Buenos Matches', img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()
