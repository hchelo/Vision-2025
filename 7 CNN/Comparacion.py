import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# ==== Configuración ====
image_size = (160, 160)
num_classes = 2
dataset_dir = 'horse-or-human'
horses_dir = os.path.join(dataset_dir, 'horses')
humans_dir = os.path.join(dataset_dir, 'humans')

# ==== Función para cargar imágenes para ambos modelos ====
def load_images_for_both_models(directory, label, image_size):
    imgs_cnn = []
    imgs_mb = []
    labels = []
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        img = image.load_img(img_path, target_size=image_size)
        img_array = image.img_to_array(img)
        imgs_cnn.append(img_array / 255.0)  # Para CNN
        imgs_mb.append(preprocess_input(np.copy(img_array)))  # Para MobileNet
        labels.append(label)
    return imgs_cnn, imgs_mb, labels

# ==== Cargar imágenes ====
horses_cnn, horses_mb, horses_labels = load_images_for_both_models(horses_dir, 1, image_size)
humans_cnn, humans_mb, humans_labels = load_images_for_both_models(humans_dir, 0, image_size)

x_cnn = np.array(horses_cnn + humans_cnn)
x_mb = np.array(horses_mb + humans_mb)
labels = to_categorical(np.array(horses_labels + humans_labels), num_classes)

# ==== Dividir dataset ====
x_train_cnn, x_test_cnn, y_train, y_test = train_test_split(x_cnn, labels, test_size=0.2, random_state=42)
x_train_mb, x_test_mb, _, _ = train_test_split(x_mb, labels, test_size=0.2, random_state=42)

# ==== MODELO CNN ====
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(160, 160, 3), padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')
])
cnn_model.load_weights('model.weights.h5')
print(" Pesos del modelo CNN cargados.")

# ==== Evaluación CNN ====
cnn_preds = cnn_model.predict(x_test_cnn)
y_true_cnn = np.argmax(y_test, axis=1)
y_pred_cnn = np.argmax(cnn_preds, axis=1)

cm_cnn = confusion_matrix(y_true_cnn, y_pred_cnn)
disp_cnn = ConfusionMatrixDisplay(confusion_matrix=cm_cnn, display_labels=["Humano", "Caballo"])
disp_cnn.plot(cmap=plt.cm.Oranges)
plt.title("Matriz de Confusión - CNN")
plt.show()

# ==== MODELO MobileNetV2 ====
base_model = MobileNetV2(input_shape=(160, 160, 3), include_top=False, weights='imagenet')
base_model.trainable = False

inputs = Input(shape=(160, 160, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
outputs = Dense(num_classes, activation='softmax')(x)
mobilenet_model = Model(inputs, outputs)
mobilenet_model.load_weights('mobilenetv2.weights.h5')
print("Pesos del modelo MobileNetV2 cargados.")

# ==== Evaluación MobileNetV2 ====
mobilenet_preds = mobilenet_model.predict(x_test_mb)
y_true_mb = np.argmax(y_test, axis=1)
y_pred_mb = np.argmax(mobilenet_preds, axis=1)

cm_mb = confusion_matrix(y_true_mb, y_pred_mb)
disp_mb = ConfusionMatrixDisplay(confusion_matrix=cm_mb, display_labels=["Humano", "Caballo"])
disp_mb.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusion - MobileNetV2")
plt.show()

# ==== Curva ROC comparativa ====
fpr_cnn, tpr_cnn, _ = roc_curve(y_true_cnn, cnn_preds[:, 1])
roc_auc_cnn = auc(fpr_cnn, tpr_cnn)

fpr_mb, tpr_mb, _ = roc_curve(y_true_mb, mobilenet_preds[:, 1])
roc_auc_mb = auc(fpr_mb, tpr_mb)

plt.figure(figsize=(8, 6))
plt.plot(fpr_cnn, tpr_cnn, label=f"CNN AUC = {roc_auc_cnn:.2f}", color='orange')
plt.plot(fpr_mb, tpr_mb, label=f"MobileNetV2 AUC = {roc_auc_mb:.2f}", color='blue')
plt.plot([0, 1], [0, 1], 'k--', label='Aleatorio')
plt.xlabel("Tasa de Falsos Positivos (FPR)")
plt.ylabel("Tasa de Verdaderos Positivos (TPR)")
plt.title("Curva ROC - Comparacion de Modelos")
plt.legend(loc='lower right')
plt.grid()
plt.show()
