import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

# Dimensiones de las imágenes
Alto, Ancho = 120, 160

# Definir el modelo igual que el de entrenamiento
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(Alto, Ancho, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))  # 5 clases, softmax

# Cargar pesos entrenados
model.load_weights('./modelo/pesos1_dc.h5')

# Mapa de etiquetas (ajustar si tus nombres de carpetas son distintos)
label_map = {
    0: "izquierda",
    1: "semi-izquierda",
    2: "adelante",
    3: "semi-derecha",
    4: "derecha"
}

from keras.preprocessing import image

# Función para predecir una imagen
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(Alto, Ancho))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    return np.argmax(pred), pred[0]

# Mostrar imágenes con predicción
root_dir = 'BD_New_DKC/testing_dkc/'

for class_folder in sorted(os.listdir(root_dir)):
    folder_path = os.path.join(root_dir, class_folder)
    if not os.path.isdir(folder_path):
        continue

    for img_file in sorted(os.listdir(folder_path)):
        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(folder_path, img_file)
        predicted_class, probabilities = predict_image(img_path)

        img = image.load_img(img_path)
        plt.imshow(img)
        plt.axis('off')

        expected_label = label_map[int(class_folder)]
        predicted_label = label_map[predicted_class]
        prob_str = ", ".join([f"{label_map[i]}: {probabilities[i]:.2f}" for i in range(5)])

        plt.title(f"Esperada: {expected_label} | Predicha: {predicted_label}\n{prob_str}", fontsize=10)
        plt.tight_layout()
        plt.pause(0.05)
        plt.clf()
