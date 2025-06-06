import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # (esto es para menos verbosidad, opcional)
os.environ['TERM'] = 'xterm-color'

# Definir el path de la base de datos
dataset_dir = 'horse-or-human'  # Cambia esto con la ruta a tu base de datos
horses_dir = os.path.join(dataset_dir, 'horses')
humans_dir = os.path.join(dataset_dir, 'humans')

# Configuración
image_size = (160,160)  # Tamaño al que redimensionar las imágenes
num_classes = 2  # Dos clases: horses y humans
batch_size = 32  # Tamaño de batch

# Función para cargar las imágenes y sus etiquetas
def load_images_from_directory(directory, label, image_size):
    images = []
    labels = []
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        img = image.load_img(img_path, target_size=image_size)  # Cargar la imagen y redimensionarla
        img_array = image.img_to_array(img) / 255.0  # Convertir a array y normalizar a [0, 1]
        images.append(img_array)
        labels.append(label)  # Añadir la etiqueta correspondiente (0 para horses, 1 para humans)
    return images, labels

# Cargar imágenes
horses_images, horses_labels = load_images_from_directory(horses_dir, 1, image_size)
humans_images, humans_labels = load_images_from_directory(humans_dir, 0, image_size)

# Combinar las imágenes de ambos directorios
images = np.array(horses_images + humans_images)
labels = np.array(horses_labels + humans_labels)

# Convertir las etiquetas a one-hot encoding
labels = to_categorical(labels, num_classes)

# Dividir en conjunto de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Definir la CNN
model = Sequential()

# Primera capa convolucional
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(160,160, 3), padding='same'))
model.add(layers.MaxPooling2D((2, 2)))

# Segunda capa convolucional
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))

# Tercera capa convolucional
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))

# Aplanar los resultados antes de pasarlos por las capas densas
model.add(layers.Flatten())

# Capa densa
model.add(layers.Dense(128, activation='relu'))

# Capa de salida
model.add(layers.Dense(num_classes, activation='softmax'))  # softmax para clasificación multiclase

# Resumen del modelo
model.summary()

# Compilar el modelo
model.compile(optimizer=Adam(), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Entrenamiento del modelo
history = model.fit(x_train, y_train, epochs=10, batch_size=batch_size, validation_split=0.2, verbose=1)

# Evaluar el modelo en el conjunto de prueba
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")

# Guardar los pesos del modelo
weights_path = 'model.weights.h5'  # Especifica la ruta del archivo donde quieres guardar los pesos
model.save_weights(weights_path)
print(f"Pesos guardados en {weights_path}")

# Hacer predicciones en las imágenes de prueba
predictions = model.predict(x_test)

# Convertir las predicciones a clases
predicted_labels = np.argmax(predictions, axis=1)

# Visualizar algunas imágenes y sus predicciones
horses_indices = np.where(np.argmax(y_test, axis=1) == 0)[0]
humans_indices = np.where(np.argmax(y_test, axis=1) == 1)[0]

# Seleccionar 5 imágenes aleatorias de horses y humans
random_horses = np.random.choice(horses_indices, 5, replace=False)
random_humans = np.random.choice(humans_indices, 5, replace=False)

# Combinar los índices aleatorios de horses y humans
random_indices = np.concatenate([random_horses, random_humans])

# Visualizar las imágenes seleccionadas y sus predicciones
for i in range(10):
    index = random_indices[i]
    plt.figure(figsize=(5,5))
    plt.imshow(x_test[index])
    
    # Obtener la predicción y la etiqueta real
    predicted_label = predicted_labels[index]
    true_label = np.argmax(y_test[index])
    
    # Mostrar el título con la predicción y la etiqueta real
    plt.title(f"Predicción: {'Caballo' if predicted_label == 1 else 'Humano'} - Real: {'Caballo' if true_label == 1 else 'Humano'}")
    plt.axis('off')  # Desactivar los ejes
    plt.show()

