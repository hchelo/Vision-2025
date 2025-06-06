import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# Configuraciones
image_size = (160, 160)
num_classes = 2
dataset_dir = 'pesados'
beetles_dir = os.path.join(dataset_dir, 'camiones')
willys_dir = os.path.join(dataset_dir, 'pickup')

# Función para cargar imágenes
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

# Cargar datos
beetles_images, beetles_labels = load_images_from_directory(beetles_dir, 0, image_size)
willys_images, willys_labels = load_images_from_directory(willys_dir, 1, image_size)

images = np.array(beetles_images + willys_images)
labels = to_categorical(np.array(beetles_labels + willys_labels), num_classes)

# Dividir en train y test
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Reconstruir el modelo
model = Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(160, 160, 3), padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Cargar pesos
model.load_weights('modelbeatle.weights.h5')

# Predicción
predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test, axis=1)

# Matriz de confusión
cm = confusion_matrix(true_labels, predicted_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Beetle", "Willys"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de confusión")
plt.show()

# Mostrar imágenes mal clasificadas
incorrect_indices = np.where(predicted_labels != true_labels)[0]
print(f"Total de imagenes mal clasificadas: {len(incorrect_indices)}")

for i in incorrect_indices:
    plt.figure(figsize=(4, 4))
    plt.imshow(x_test[i])
    pred = predicted_labels[i]
    true = true_labels[i]
    plt.title(f"Predicho: {'Willys' if pred == 1 else 'Beetle'} | Real: {'Willys' if true == 1 else 'Beetle'}")
    plt.axis('off')
    plt.show()
