import cv2
import mediapipe as mp

# Inicializa MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Carga la imagen
image_path = "dedos/5_10.png"  # Cambia esto con tu imagen
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Procesa la imagen con MediaPipe Hands
with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
    results = hands.process(image_rgb)

    # Verifica si se detectaron manos
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Dibuja los puntos de las manos sobre la imagen original
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(50,255,0),thickness=4,circle_radius=5),
                mp_drawing.DrawingSpec(color=(0,50,50),thickness=3))

# Muestra la imagen con las manos detectadas
cv2.imshow("Manos detectadas", image)
cv2.waitKey(0)
cv2.destroyAllWindows()