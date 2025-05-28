import numpy as np

# Leer el archivo y cargar los valores RGB únicos
valores_rgb = []

with open('valores_rgb_unicos.txt', 'r') as f:
    for line in f:
        # Leer los valores de cada línea (R, G, B) y convertirlos a enteros
        rgb = tuple(map(int, line.strip().split(', ')))
        valores_rgb.append(rgb)

# Convertir la lista de valores RGB en una matriz numpy
MatrizRGB = np.array(valores_rgb)

# Calcular la media de cada canal (R, G, B)
media_1 = np.mean(MatrizRGB, axis=0)

# Calcular la desviación estándar de cada canal (R, G, B)
desviacion_1 = np.std(MatrizRGB, axis=0)

# Calcular la matriz de covarianza
covarianza_1 = np.cov(MatrizRGB, rowvar=False)

# Crear la matriz de covarianza diagonalizada
cova_1 = np.diag(np.diag(covarianza_1))

# Calcular el determinante y la inversa de la matriz de covarianza
deter_1 = np.linalg.det(cova_1)
inversa_1 = np.linalg.inv(cova_1)

# Definir un pixel específico [R, G, B]
R, G, B = 100, 150, 200  # Aquí puedes poner el valor de cualquier pixel

# Crear el vector del pixel
pixel = np.array([R, G, B], dtype=np.float64)

# Imprimir los resultados
print(f"Media por canal: {media_1}")
print(f"Desviación estándar por canal: {desviacion_1}")
print(f"Matriz de covarianza diagonalizada: {cova_1}")
print(f"Determinante de la matriz de covarianza: {deter_1}")
print(f"Inversa de la matriz de covarianza: \n{inversa_1}")
print(f"Valor del pixel: {pixel}")
