import cv2
import os

# Ruta completa de la imagen
imagen_path = r'Tomates.jpg'  # Actualiza esta línea con la ruta correcta

# Leer la imagen
img = cv2.imread(imagen_path)

# Verificar si la imagen fue cargada correctamente
if img is None:
    print("Error al cargar la imagen.")
else:
    height, width, _ = img.shape

    # Aplicar el filtro de detección de piel
    for y in range(height):
        for x in range(width):
            b, g, r = img[y, x]
            if (r >= 169) and (g >= 0)  and (b >= 0) and (r <= 242) and (g <= 135)  and (b <= 46):
                img[y, x] = (0, 0, 255)
            else:
                #continue
                img[y, x] = (0, 0, 0)  # Cambiar el color a negro si no cumple con los criterios

    # Mostrar la imagen resultante
    cv2.imshow('Imagen Filtrada - tomate.jpg', img)
    cv2.waitKey(0)  # Espera hasta que se presione una tecla
    cv2.destroyAllWindows()
