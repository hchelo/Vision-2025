import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# Ruta al modelo
MODEL_PATH = "models/efficientdet_lite0_int8.tflite"
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

# Configuración de visualización
MARGIN = 10
ROW_SIZE = 10
FONT_SIZE = 1
FONT_THICKNESS = 2
TEXT_COLOR_PERSON = (0, 255, 0)  # verde para personas

def visualize_persons(image: np.ndarray, detection_result) -> np.ndarray:
    """Dibuja las cajas y etiquetas SOLO para personas."""
    for detection in detection_result.detections:
        category = detection.categories[0]
        if category.category_name == "person":
            bbox = detection.bounding_box
            start_point = int(bbox.origin_x), int(bbox.origin_y)
            end_point = int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height)
            cv2.rectangle(image, start_point, end_point, TEXT_COLOR_PERSON, 2)

            probability = round(category.score, 2)
            label = f"Person ({probability})"
            text_location = (start_point[0], max(0, start_point[1] - 10))
            cv2.putText(image, label, text_location, cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, TEXT_COLOR_PERSON, FONT_THICKNESS)
    return image

video_path = "cocacola.mp4"  # o usa 0 para webcam
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise FileNotFoundError(f"No se pudo abrir el video: {video_path}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    start_time = time.time()
    detection_result = detector.detect(mp_frame)
    inference_time = (time.time() - start_time) * 1000

    annotated_frame = visualize_persons(frame, detection_result)

    fps_text = f"Inference Time: {inference_time:.2f} ms"
    cv2.putText(annotated_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, TEXT_COLOR_PERSON, FONT_THICKNESS)

    cv2.imshow("Detección SOLO personas", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
