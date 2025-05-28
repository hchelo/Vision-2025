import cv2
import numpy as np

# Variables globales para almacenar los puntos seleccionados
puntos = []

# Función para manejar la selección de puntos con el mouse
def seleccionar_puntos(event, x, y, flags, param):
    global puntos

    if event == cv2.EVENT_LBUTTONDOWN:
        # Añadir punto
        puntos.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)  # Dibujar un círculo rojo
        if len(puntos) > 1:
            # Dibujar el polígono
            cv2.polylines(img, [np.array(puntos, np.int32)], isClosed=False, color=(0, 255, 0), thickness=2)
        cv2.imshow('Imagen', img)

# Ruta completa de la imagen
imagen_path = r'Tomates.jpg'  # Actualiza esta línea con la ruta correcta

# Leer la imagen
img = cv2.imread(imagen_path)

# Crear una ventana para mostrar la imagen
cv2.imshow('Imagen', img)
cv2.setMouseCallback('Imagen', seleccionar_puntos)

# Esperar hasta que se hagan 10 puntos
while len(puntos) < 10:
    cv2.waitKey(1)

# Una vez que se han seleccionado los puntos, cortar la imagen
if len(puntos) == 10:
    # Convertir la lista de puntos en un array de numpy
    puntos_np = np.array(puntos, np.int32)
    
    # Crear una máscara (blanco y negro) para el área del polígono
    mascara = np.zeros_like(img)
    cv2.fillPoly(mascara, [puntos_np], (255, 255, 255))

    # Hacer un AND entre la imagen original y la máscara
    imagen_recortada = cv2.bitwise_and(img, mascara)

    # Mostrar y guardar la imagen recortada
    cv2.imshow('Imagen Recortada', imagen_recortada)
    cv2.imwrite('Imagen_Recortada.jpg', imagen_recortada)

# Esperar hasta que se presione una tecla para cerrar
cv2.waitKey(0)
cv2.destroyAllWindows()
