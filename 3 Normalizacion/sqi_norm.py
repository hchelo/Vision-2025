import cv2
import numpy as np
import matplotlib.pyplot as plt

def self_quotient_image(image, filter_size=30, epsilon=1e-6):
    # Asegurarse de que filter_size es impar
    if filter_size <= 0 or filter_size % 2 == 0:
        filter_size = max(3, filter_size | 1)  # Convertir a un número impar positivo mínimo 3

    # Suavizado de la imagen con un filtro gaussiano
    smoothed = cv2.GaussianBlur(image, (filter_size, filter_size), 0)

    # Cálculo de la Self-Quotient Image
    sqi = image / (smoothed + epsilon)

    # Normalización para que la imagen esté en un rango de 0 a 255
    sqi_normalized = cv2.normalize(sqi, None, 0, 255, cv2.NORM_MINMAX)

    return sqi_normalized.astype(np.uint8)

# Cargar imagen de entrada (escala de grises)
image_path = "yale4.bmp"  # Cambia esta ruta por la de tu imagen
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise ValueError("No se pudo cargar la imagen. Asegúrate de que la ruta sea correcta.")

# Aplicar el método Self-Quotient Image
sqi_result = self_quotient_image(image, filter_size=30)

# Mostrar resultados
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Imagen Original")
plt.imshow(image, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Self-Quotient Image")
plt.imshow(sqi_result, cmap="gray")
plt.axis("off")

plt.show()
