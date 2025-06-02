import cv2
import numpy as np

# Cargar la imagen base y convertir a escala de grises
img1 = cv2.imread('cocacola.jpg')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray1 = cv2.equalizeHist(gray1)

# Crear detector ORB
orb = cv2.ORB_create(nfeatures=1500)

# Detectar keypoints y descriptores en la imagen base
kp1, des1 = orb.detectAndCompute(gray1, None)

# Cargar el video
cap = cv2.VideoCapture('cocacola.mp4')

# Matcher para descriptores binarios
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# Empieza desde el frame 100
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
count_frame = 100

while True:
    ret, frame = cap.read()
    if not ret:
        break

    count_frame += 1
    if count_frame % 1 != 0:
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
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) > 18:
        # Obtener puntos clave emparejados
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Calcular la homografía
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if M is not None:
            h, w = gray1.shape
            pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            # Dibujar el encuadre (cuadro verde)
            frame = cv2.polylines(frame, [np.int32(dst)], True, (255, 250, 0), 3, cv2.LINE_AA)

    # Mostrar el frame con o sin encuadre
    cv2.imshow('ORB Encuadre con Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
