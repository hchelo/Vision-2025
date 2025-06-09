from ultralytics import YOLO

# Cargar el modelo YOLOv8 preentrenado (puedes usar yolov8n.pt, yolov8s.pt, etc.)
model = YOLO('yolov8n.pt')

# Imprimir todas las clases y sus IDs
print("Clases del modelo YOLOv8:")
for class_id, class_name in model.names.items():
    print(f"{class_id:2}: {class_name}")
