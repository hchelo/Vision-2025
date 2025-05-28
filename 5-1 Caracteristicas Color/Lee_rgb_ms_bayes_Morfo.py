import cv2
import numpy as np

# Ruta completa de la imagen
imagen_path = 'Tomates.jpg'  # Actualiza con la ruta correcta

# Leer la imagen
imagen = cv2.imread(imagen_path)

# Convertir la imagen a formato flotante para cálculos precisos
imagen_float = np.float64(imagen)

# Leer los valores de media, desviación estándar y matriz de covarianza desde el archivo
valores_rgb = []
with open('valores_rgb_unicos.txt', 'r') as f:
    for line in f:
        rgb = tuple(map(int, line.strip().split(', ')))
        valores_rgb.append(rgb)

# Convertir la lista de valores RGB en una matriz numpy
MatrizRGB = np.array(valores_rgb)

# Calcular la media de cada canal (R, G, B)
media_1 = np.mean(MatrizRGB, axis=0)

# Calcular la matriz de covarianza diagonalizada
covarianza_1 = np.cov(MatrizRGB, rowvar=False)
cova_1 = np.diag(np.diag(covarianza_1))

# Calcular el determinante e inversa de la matriz de covarianza
deter_1 = np.linalg.det(cova_1)
inversa_1 = np.linalg.inv(cova_1)

# Vectorizar el cálculo de la distancia de Mahalanobis
imagen_rgb_float = imagen_float.reshape((-1, 3))  
dif_pixel_media = imagen_rgb_float - media_1  
Sk_1 = np.sum(dif_pixel_media @ inversa_1 * dif_pixel_media, axis=1)  

# Calcular Pk_1 para todos los píxeles
constante_Pk_1 = 1 / np.sqrt(((2 * np.pi) ** 3) * deter_1)
Pk_1 = constante_Pk_1 * np.exp(-Sk_1 / 2)
umbral = 0.00000005

# Crear una máscara binaria (blanco = área detectada)
mask = np.zeros((imagen.shape[0], imagen.shape[1]), dtype=np.uint8)
mask[Pk_1.reshape(imagen.shape[:2]) > umbral] = 255  # Regiones blancas donde se detecta el objeto

# Definir un kernel de 5x5 para operaciones morfológicas
kernel1 = np.ones((15, 15), np.uint8)
kernel2 = np.ones((15, 15), np.uint8)

# Aplicar dilatación
mask_erode = cv2.erode(mask, kernel1, iterations=1)

mask_dilate = cv2.dilate(mask_erode, kernel1, iterations=2)


# Aplicar apertura (erosión seguida de dilatación)
#mask_opened = cv2.morphologyEx(mask_dilated, cv2.MORPH_CLOSE, kernel1)

# Aplicar la máscara procesada a la imagen original para mantener el color rojo
new_RGB_filtered = cv2.bitwise_and(imagen, imagen, mask=mask_dilate)

# Mostrar la imagen original
cv2.imshow('Imagen Original', imagen)

# Mostrar la imagen filtrada antes de la morfología
cv2.imshow('Imagen Filtrada', cv2.bitwise_and(imagen, imagen, mask=mask))

# Mostrar la imagen después de la dilatación
cv2.imshow('Erosionada', cv2.bitwise_and(imagen, imagen, mask=mask_erode))

# Mostrar la imagen después de la apertura
cv2.imshow('Cierre', new_RGB_filtered)

# Esperar hasta que se presione una tecla para cerrar
cv2.waitKey(0)
cv2.destroyAllWindows()
