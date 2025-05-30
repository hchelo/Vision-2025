import cv2
import numpy as np

# Cargar la imagen y convertirla a escala de grises
img = cv2.imread('fanta.jpg')
factor_de_escala = 0.3
ancho_nuevo = int(img.shape[1] * factor_de_escala)
alto_nuevo = int(img.shape[0] * factor_de_escala)
img = cv2.resize(img, (ancho_nuevo, alto_nuevo), interpolation=cv2.INTER_AREA)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Crear el objeto SIFT
sift = cv2.SIFT_create()

# Detectar keypoints y descriptores en la imagen base
kp_img, des_img = sift.detectAndCompute(gray_img, None)

# Crear objeto VideoCapture
cap = cv2.VideoCapture('fanta3.mp4')

# Matcher
bf = cv2.BFMatcher()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_frame, des_frame = sift.detectAndCompute(gray_frame, None)

    if des_frame is None or len(des_frame) < 2:
        continue

    matches = bf.knnMatch(des_img, des_frame, k=2)

    # Ratio test de Lowe
    good_matches = []
    for m, n in matches:
        if m.distance < 0.85 * n.distance:
            good_matches.append(m)

    if len(good_matches) > 10:
        src_pts = np.float32([kp_img[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Calcular la homografía
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if M is not None:
            h, w = gray_img.shape
            pts = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts, M)

            # Dibujar el encuadre (cuadrado verde)
            frame = cv2.polylines(frame, [np.int32(dst)], True, (0,255,0), 3, cv2.LINE_AA)

    # Mostrar el resultado
    cv2.imshow('SIFT Encuadre', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
