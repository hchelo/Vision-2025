import cv2
import numpy as np

# Ruta completa de la imagen
imagen_path = 'Tomates.jpg'  # Actualiza esta línea con la ruta correcta

# Leer la imagen
imagen = cv2.imread(imagen_path)

# Convertir la imagen a formato flotante para cálculos precisos
imagen_float = np.float64(imagen)

# Leer los valores de media, desviación estándar y matriz de covarianza desde el archivo 'valores_rgb_unicos.txt'
valores_rgb = []

with open('valores_rgb_unicos.txt', 'r') as f:
    for line in f:
        rgb = tuple(map(int, line.strip().split(', ')))
        valores_rgb.append(rgb)

# Convertir la lista de valores RGB en una matriz numpy
MatrizRGB = np.array(valores_rgb)

# Calcular la media de cada canal (R, G, B)
media_1 = np.mean(MatrizRGB, axis=0)

# Calcular la matriz de covarianza
covarianza_1 = np.cov(MatrizRGB, rowvar=False)

# Crear la matriz de covarianza diagonalizada
cova_1 = np.diag(np.diag(covarianza_1))

# Calcular el determinante de la matriz de covarianza
deter_1 = np.linalg.det(cova_1)

# Calcular la inversa de la matriz de covarianza
inversa_1 = np.linalg.inv(cova_1)

# Inicializar una nueva imagen para almacenar los píxeles filtrados
new_RGB = np.zeros_like(imagen, dtype=np.uint8)

# Recorrer la imagen pixel por pixel
for y in range(imagen.shape[0]):  # Recorrer filas (y)
    for x in range(imagen.shape[1]):  # Recorrer columnas (x)
        # Obtener el valor RGB del píxel actual
        PixelRGB = imagen[y, x]
        
        # Calcular el valor de Sk_1
        Sk_1 = (PixelRGB - media_1) @ inversa_1 @ (PixelRGB - media_1).T
        
        # Calcular el valor de Pk_1
        Pk_1 = (1 / np.sqrt(((2 * np.pi) ** 3) * deter_1)) * np.exp(-Sk_1 / 2)
        
        # Filtrar el píxel según el valor de Pk_1
        if Pk_1 > 0.00000001:
            new_RGB[y, x] = [0, 0, 255]  # Asignar color rojo
        else:
            new_RGB[y, x] = [0, 0, 0]  # Asignar color negro

# Mostrar la imagen original
cv2.imshow('Imagen Original', imagen)

# Mostrar la nueva imagen filtrada
cv2.imshow('Imagen Filtrada', new_RGB)

# Esperar hasta que se presione una tecla para cerrar
cv2.waitKey(0)
cv2.destroyAllWindows()
