import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# Configuración
image_size = (160, 160)
num_classes = 2
dataset_dir = 'pesados'
horses_dir = os.path.join(dataset_dir, 'camiones')
humans_dir = os.path.join(dataset_dir, 'pickup')

# Función para cargar imágenes
def load_images_from_directory(directory, label, image_size):
    images = []
    labels = []
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        img = image.load_img(img_path, target_size=image_size)
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        images.append(img_array)
        labels.append(label)
    return images, labels

# Cargar imágenes
horses_images, horses_labels = load_images_from_directory(horses_dir, 1, image_size)
humans_images, humans_labels = load_images_from_directory(humans_dir, 0, image_size)

images = np.array(horses_images + humans_images)
labels = to_categorical(np.array(horses_labels + humans_labels), num_classes)

# Dividir dataset
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Reconstruir el modelo con la misma arquitectura
base_model = MobileNetV2(input_shape=(160, 160, 3), include_top=False, weights='imagenet')
base_model.trainable = False

inputs = Input(shape=(160, 160, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
outputs = Dense(num_classes, activation='softmax')(x)
model = Model(inputs, outputs)

# Compilar
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Cargar los pesos guardados
model.load_weights('mobilenetv2.weights.h5')
print("Pesos cargados desde 'mobilenetv2.weights.h5'")

# Evaluar y mostrar matriz de confusión
loss, acc = model.evaluate(x_test, y_test)
print(f"Precisión después de cargar pesos: {acc:.4f}")

# ======= Curva ROC =======
predictions = model.predict(x_test)

# Obtener las probabilidades predichas para la clase "Caballo" (índice 1)
fpr, tpr, _ = roc_curve(np.argmax(y_test, axis=1), predictions[:, 1])
roc_auc = auc(fpr, tpr)

# Graficar la curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'MobileNetV2 AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--', label='Aleatorio')
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('Curva ROC - MobileNetV2')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# ======= Matriz de Confusión =======
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test, axis=1)

cm = confusion_matrix(true_labels, predicted_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Humano", "Caballo"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de confusion (modelo cargado)")
plt.show()

# Mostrar imágenes mal clasificadas
incorrect_indices = np.where(predicted_labels != true_labels)[0]
print(f"Total mal clasificadas: {len(incorrect_indices)}")

for i in incorrect_indices:
    plt.figure(figsize=(4, 4))
    plt.imshow((x_test[i] + 1) * 127.5 / 255.0)  # Desnormalizar para visualización
    pred = predicted_labels[i]
    true = true_labels[i]
    plt.title(f"Predicho: {'Caballo' if pred == 1 else 'Humano'} | Real: {'Caballo' if true == 1 else 'Humano'}")
    plt.axis('off')
    plt.show()
