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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Configuración
image_size = (160, 160)  # MobileNetV2 espera 160x160 o más
num_classes = 2
dataset_dir = 'horse-or-human'
horses_dir = os.path.join(dataset_dir, 'horses')
humans_dir = os.path.join(dataset_dir, 'humans')

# Cargar imágenes y etiquetas
def load_images_from_directory(directory, label, image_size):
    images = []
    labels = []
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        img = image.load_img(img_path, target_size=image_size)
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)  # MobileNetV2 necesita preprocesamiento especial
        images.append(img_array)
        labels.append(label)
    return images, labels

horses_images, horses_labels = load_images_from_directory(horses_dir, 1, image_size)
humans_images, humans_labels = load_images_from_directory(humans_dir, 0, image_size)

images = np.array(horses_images + humans_images)
labels = to_categorical(np.array(horses_labels + humans_labels), num_classes)

# Dividir en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Cargar MobileNetV2 sin la parte final (head)
base_model = MobileNetV2(input_shape=(160, 160, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Congelar la base para entrenamiento inicial

# Agregar nuevas capas densas (head personalizado)
inputs = Input(shape=(160, 160, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
outputs = Dense(num_classes, activation='softmax')(x)
model = Model(inputs, outputs)

# Compilar y entrenar
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Entrenar modelo (puedes ajustar epochs)
history = model.fit(x_train, y_train, epochs=10, validation_split=0.2, batch_size=32)

# Evaluación
loss, acc = model.evaluate(x_test, y_test)
print(f"Precisión en test: {acc:.4f}")

# Matriz de confusión
predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test, axis=1)

cm = confusion_matrix(true_labels, predicted_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Humano", "Caballo"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de confusión con MobileNetV2")
plt.show()

# Mostrar imágenes mal clasificadas
incorrect_indices = np.where(predicted_labels != true_labels)[0]
print(f"Total mal clasificadas: {len(incorrect_indices)}")

# Guardar solo los pesos (forma correcta)
model.save_weights('mobilenetv2.weights.h5')
print("Pesos guardados en 'mobilenetv2.weights.h5'")

for i in incorrect_indices:
    plt.figure(figsize=(4, 4))
    plt.imshow((x_test[i] + 1) * 127.5 / 255.0)  # Desnormalizar para visualización
    pred = predicted_labels[i]
    true = true_labels[i]
    plt.title(f"❌ Predicho: {'Caballo' if pred == 1 else 'Humano'} | Real: {'Caballo' if true == 1 else 'Humano'}")
    plt.axis('off')
    plt.show()
