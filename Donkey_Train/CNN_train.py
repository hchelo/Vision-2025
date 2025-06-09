import tensorflow as tf
import os
import time
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Funciones auxiliares para medir el tiempo
def TicTocGenerator():
    ti = 0
    tf = time.time()
    while True:
        ti = tf
        tf = time.time()
        yield tf - ti

TicToc = TicTocGenerator()

def toc(tempBool=True):
    tempTimeInterval = next(TicToc)
    if tempBool:
        print("Tiempo transcurrido: %f segundos.\n" % tempTimeInterval)

def tic():
    toc(False)

# Dimensiones de las imágenes
Alto, Ancho = 120, 160

# Construcción del modelo CNN desde cero
model = Sequential()

# Bloque 1
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(Alto, Ancho, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Bloque 2
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Bloque 3
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Aplanamiento y capas densas
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))  # 5 clases

# Compilar el modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Aumento de datos para entrenamiento
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.3,
    #horizontal_flip=True,
    fill_mode='nearest'
)

# Datos de validación sin aumento, solo reescalado
test_datagen = ImageDataGenerator(rescale=1./255)

# Cargar conjuntos de datos
training_set = train_datagen.flow_from_directory(
    'BD_New_DKC/training_dkc',
    target_size=(Alto, Ancho),
    batch_size=16,
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    'BD_New_DKC/testing_dkc',
    target_size=(Alto, Ancho),
    batch_size=16,
    class_mode='categorical'
)

# Callbacks para control de entrenamiento
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

print("Entrenando el modelo...")
start_time = time.time()

model.fit(
    training_set,
    steps_per_epoch=len(training_set),
    epochs=150,
    validation_data=test_set,
    validation_steps=len(test_set),
    callbacks=[early_stopping, reduce_lr, checkpoint]
)

end_time = time.time()
print(f"Tiempo total de entrenamiento: {end_time - start_time:.2f} segundos.")

# Guardar modelo y pesos
target_dir = './modelo/'
if not os.path.exists(target_dir):
    os.mkdir(target_dir)
model.save(os.path.join(target_dir, 'modelo1_VC.h5'))
model.save_weights(os.path.join(target_dir, 'pesos1_VC.h5'))

# Función para predecir una sola imagen
def predict_image(image_path):
    from keras.preprocessing import image
    img = image.load_img(image_path, target_size=(Alto, Ancho))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0  # Normalizar igual que durante el entrenamiento
    pred = model.predict(img)
    return np.argmax(pred[0])

# Ejemplo de prueba con imágenes específicas
test_images = [
    ('BD_New_DKC/testing_dkc/0/izq0117.png', 0),
    ('BD_New_DKC/testing_dkc/0/izq0147.png', 0),
    ('BD_New_DKC/testing_dkc/1/semizq 0117.png', 1),
    ('BD_New_DKC/testing_dkc/1/semizq 0136.png', 1),
    ('BD_New_DKC/testing_dkc/2/adelante0081.png', 2),
    ('BD_New_DKC/testing_dkc/2/adelante0099.png', 2),
    ('BD_New_DKC/testing_dkc/3/semder0072.png', 3),
    ('BD_New_DKC/testing_dkc/3/semder0081.png', 3),
    ('BD_New_DKC/testing_dkc/4/der0187.png', 4),
    ('BD_New_DKC/testing_dkc/4/der0194.png', 4)
]

for path, expected in test_images:
    print(f"Probando imagen: {path} (Esperado: {expected})")
    tic()
    result = predict_image(path)
    print(f"Resultado: {result}")
    toc()
