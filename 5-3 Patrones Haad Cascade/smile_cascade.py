import cv2

# Cargar clasificadores Haar
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Leer imagen
image_path = "3views.jpg"  # Cambia por la ruta de tu imagen
image = cv2.imread(image_path)

if image is None:
    print(f"No se pudo cargar la imagen: {image_path}")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detectar rostros
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
    face_gray = gray[y:y+h, x:x+w]
    face_color = image[y:y+h, x:x+w]

    # Detectar sonrisa dentro del rostro
    smiles = smile_cascade.detectMultiScale(
        face_gray,
        scaleFactor=1.8,
        minNeighbors=6
    )

    if len(smiles) > 0:
        # Si hay sonrisa → cuadro verde
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, "Alegre", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    else:
        # Si no hay sonrisa → cuadro rojo
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(image, "Serio", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

# Mostrar la imagen procesada
cv2.imshow("Resultado", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
