# Red Neuronal Convolucional
import tensorflow as tf
import os
import time
import numpy as np
# Importar las bibliotecas y paquetes de Keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image 
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


# Funciones auxiliares para medir el tiempo
def TicTocGenerator():
    # Generador que devuelve diferencias de tiempo
    ti = 0           # tiempo inicial
    tf = time.time() # tiempo final
    while True:
        ti = tf
        tf = time.time()
        yield tf - ti # devuelve la diferencia de tiempo

TicToc = TicTocGenerator() # crear una instancia del generador TicTocGen

def toc(tempBool=True):
    # Imprime la diferencia de tiempo generada por la instancia TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print("Tiempo transcurrido: %f segundos.\n" % tempTimeInterval)

def tic():
    # Registra un tiempo en TicToc, marca el comienzo de un intervalo de tiempo
    toc(False)

# Dimensiones de las imágenes
Alto, Ancho = 120, 160

# Inicializando la CNN
classifier = Sequential()

# Paso 1 - Convolución y Max Pooling
classifier.add(Conv2D(8, (3, 3), input_shape=(Alto, Ancho, 3), activation='relu', kernel_regularizer=l2(0.01)))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.5))  # Aumentar el dropout

# Añadiendo una segunda capa convolucional y Max Pooling
classifier.add(Conv2D(8, (3, 3), activation='relu', kernel_regularizer=l2(0.01)))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.5))  # Aumentar el dropout

# Añadiendo una tercera capa convolucional y Max Pooling
classifier.add(Conv2D(8, (3, 3), activation='relu', kernel_regularizer=l2(0.01)))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.5))  # Aumentar el dropout

# Paso 2 - Aplanamiento
classifier.add(Flatten())

# Paso 3 - Conexión completa
classifier.add(Dense(units=128, activation='relu', kernel_regularizer=l2(0.01)))  # Aumentar el número de unidades
classifier.add(Dropout(0.5))
classifier.add(Dense(units=5, activation='sigmoid', kernel_regularizer=l2(0.01)))

# Compilando la CNN
classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Parte 2 - Ajuste de la CNN a las imágenes

# Aumento de datos para el conjunto de entrenamiento
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=5,  # Nuevas transformaciones
    width_shift_range=0.2,
    height_shift_range=0.2,
    fill_mode='nearest'
)

# Aumento de datos para el conjunto de prueba
test_datagen = ImageDataGenerator(rescale=1./255)

# Conjunto de entrenamiento
training_set = train_datagen.flow_from_directory(
    'BD_New_DKC/training_dkc',
    target_size=(Alto, Ancho),
    batch_size=16,  # Ajustar el batch size
    class_mode='categorical'
)

# Conjunto de prueba
test_set = test_datagen.flow_from_directory(
    'BD_New_DKC/testing_dkc',
    target_size=(Alto, Ancho),
    batch_size=16,  # Ajustar el batch size
    class_mode='categorical'
)

# Calcular los pasos por época
steps_per_epoch = len(training_set)
validation_steps = len(test_set)

# Configurar Early Stopping, reducción de la tasa de aprendizaje y puntos de control del modelo
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

print("Entrenando el modelo...")
start_time = time.time()

classifier.fit(
    training_set,
    steps_per_epoch=steps_per_epoch,
    epochs=1000,
    validation_data=test_set,
    validation_steps=validation_steps,
    callbacks=[early_stopping, reduce_lr, checkpoint]
)
end_time = time.time()
print(f"Elapsed time: {end_time - start_time} seconds.")

# Guardar el modelo y los pesos
target_dir = './modelo/'
if not os.path.exists(target_dir):
    os.mkdir(target_dir)
classifier.save(os.path.join(target_dir, 'modelo1_dc.h5'))
classifier.save_weights(os.path.join(target_dir, 'pesos1_dc.h5'))

# Función para hacer una predicción en una sola imagen
def predict_image(image_path):
    test_image = image.load_img(image_path, target_size=(Alto, Ancho))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    array = classifier.predict(test_image)
    result = np.argmax(array[0])
    return result

# Prueba de predicciones
test_images = [
    ('BD_New_DKC/testing_dkc/0/izq0117.png', 0),
    ('BD_New_DKC/testing_dkc/0/izq0147.png', 0),
    ('BD_New_DKC/testing_dkc/1/semizq 0117.png', 0),
    ('BD_New_DKC/testing_dkc/1/semizq 0136.png', 0),
    ('BD_New_DKC/testing_dkc/2/adelante0081.png', 2),
    ('BD_New_DKC/testing_dkc/2/adelante0099.png', 2),
    ('BD_New_DKC/testing_dkc/3/semder0072.png', 3),
    ('BD_New_DKC/testing_dkc/3/semder0081.png', 3),
    ('BD_New_DKC/testing_dkc/4/der0187.png', 4),
    ('BD_New_DKC/testing_dkc/4/der0194.png', 4)
]

for image_path, expected in test_images:
    print(f"Probando imagen: {image_path} (Esperado: {expected})")
    tic()
    result = predict_image(image_path)
    print(f"Resultado: {result}")
    toc()
