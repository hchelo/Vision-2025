from ultralytics import YOLO
import cv2

# Cargar modelo YOLOv8
model = YOLO('yolov8n.pt')  # Puedes cambiar a yolov8s.pt, yolov8m.pt, etc.

# Cargar el video
cap = cv2.VideoCapture('jeepepetas.mp4')

# Verificar apertura del video
if not cap.isOpened():
    print("No se pudo abrir el video.")
    exit()

# Bucle de procesamiento
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inferencia
    results = model(frame)[0]

    # Filtrar por clase y confianza > 0.5
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        if cls_id in [2, 7] and conf > 0.5:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f'{model.names[cls_id]} {conf:.2f}'
            color = (0, 255, 0) if cls_id == 2 else (0, 0, 255)  # Verde: car, Rojo: truck
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Mostrar frame
    cv2.imshow('Detección de Autos y Camiones (conf > 0.5)', frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
