from ultralytics import YOLO
import cv2

# Cargar el modelo YOLOv8
model = YOLO('yolov8n.pt')

# Abrir video
cap = cv2.VideoCapture('jeepepetas.mp4')
if not cap.isOpened():
    print("No se pudo abrir el video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inferencia
    results = model(frame)[0]

    # Inicializar variables para guardar la mejor detección
    best_car = None
    best_truck = None

    # Buscar car (2) y truck (7)
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id == 2 and conf > 0.5:
            if not best_car or conf > best_car['conf']:
                best_car = {'conf': conf, 'box': box}
        elif cls_id == 7 and conf > 0.5:
            if not best_truck or conf > best_truck['conf']:
                best_truck = {'conf': conf, 'box': box}

    # Comparar confianza y dibujar solo uno
    if best_car and (not best_truck or best_car['conf'] >= best_truck['conf']):
        box = best_car['box']
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f"car {best_car['conf']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    elif best_truck:
        box = best_truck['box']
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f"truck {best_truck['conf']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Mostrar frame
    cv2.imshow('Detección: Solo el más confiable entre car y truck', frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
