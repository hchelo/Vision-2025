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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TERM'] = 'xterm-color'

# Definir el path de la base de datos
dataset_dir = 'pesados'
beetles_dir = os.path.join(dataset_dir, 'camiones')
willys_dir = os.path.join(dataset_dir, 'pickup')

# Configuración
image_size = (160, 160)
num_classes = 2
batch_size = 32

# Función para cargar las imágenes y sus etiquetas
def load_images_from_directory(directory, label, image_size):
    images = []
    labels = []
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        img = image.load_img(img_path, target_size=image_size)
        img_array = image.img_to_array(img) / 255.0
        images.append(img_array)
        labels.append(label)
    return images, labels

# Cargar imágenes
beetles_images, beetles_labels = load_images_from_directory(beetles_dir, 0, image_size)
willys_images, willys_labels = load_images_from_directory(willys_dir, 1, image_size)

# Combinar las imágenes de ambos directorios
images = np.array(beetles_images + willys_images)
labels = np.array(beetles_labels + willys_labels)

# Convertir las etiquetas a one-hot encoding
labels = to_categorical(labels, num_classes)

# Dividir en conjunto de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Definir la CNN
model = Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(160, 160, 3), padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))

model.summary()

# Compilar el modelo
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(x_train, y_train, epochs=10, batch_size=batch_size, validation_split=0.2, verbose=1)

# Evaluar el modelo
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# Guardar los pesos del modelo
weights_path = 'modelbeatle.weights.h5'
model.save_weights(weights_path)
print(f"Pesos guardados en {weights_path}")

# Hacer predicciones
predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)

# Visualizar algunas imágenes y sus predicciones
beetle_indices = np.where(np.argmax(y_test, axis=1) == 0)[0]
willys_indices = np.where(np.argmax(y_test, axis=1) == 1)[0]

random_beetles = np.random.choice(beetle_indices, 5, replace=False)
random_willys = np.random.choice(willys_indices, 5, replace=False)
random_indices = np.concatenate([random_beetles, random_willys])

for i in range(10):
    index = random_indices[i]
    plt.figure(figsize=(5, 5))
    plt.imshow(x_test[index])
    predicted_label = predicted_labels[index]
    true_label = np.argmax(y_test[index])
    plt.title(f"Predicción: {'Willys' if predicted_label == 1 else 'Beetle'} - Real: {'Willys' if true_label == 1 else 'Beetle'}")
    plt.axis('off')
    plt.show()
