import cv2
import numpy as np
import matplotlib.pyplot as plt

def local_normalization(image, window_size=5, epsilon=3):

    # Convertir la imagen a flotante para cálculos precisos
    image_float = image.astype(np.float32)
    
    # Calcular el promedio local (E)
    mean_image = cv2.blur(image_float, (window_size, window_size))
    
    # Calcular la desviación estándar local (s)
    sqr_image = cv2.blur(image_float**2, (window_size, window_size))
    std_image = np.sqrt(sqr_image - mean_image**2)
    
    # Normalizar la imagen localmente
    normalized_image = (image_float - mean_image) / (std_image + epsilon)
    
    # Normalizar a rango [0, 255]
    normalized_image = cv2.normalize(normalized_image, None, 0, 255, cv2.NORM_MINMAX)
    
    return normalized_image.astype(np.uint8)

# Cargar la imagen en escala de grises
image_path = "yale1.bmp"  # Cambia esta ruta por la de tu imagen
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise ValueError("No se pudo cargar la imagen. Asegúrate de que la ruta sea correcta y la imagen esté en escala de grises.")

# Aplicar la normalización local
compensated_image = local_normalization(image, window_size=5)

# Mostrar los resultados
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Imagen Original")
plt.imshow(image, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Imagen Normalizada Localmente")
plt.imshow(compensated_image, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()
