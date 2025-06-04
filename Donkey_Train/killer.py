# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
import os
import time

def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)

Alto = 64
Ancho = 64

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(24, (3, 3), input_shape = (Alto, Ancho, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(8, (3, 3), padding ="same"))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
#classifier.add(Dense(units = 1, activation = 'sigmoid'))
classifier.add(Dropout(0.5))
classifier.add(Dense(5, activation='softmax'))
# Compiling the CNN
classifier.compile(loss='categorical_crossentropy', # <== LOOK HERE!
              optimizer='adam',
              metrics=['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('BD_mov/TRAIN',
                                                 target_size = (Alto, Ancho),
                                                 batch_size = 64,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('BD_mov/TEST',
                                            target_size = (Alto, Ancho),
                                            batch_size = 64,
                                            class_mode = 'categorical')
print(".....................1............")
tic()
classifier.fit_generator(training_set,
                         steps_per_epoch = 15000,
                         epochs = 15,
                         validation_data = test_set,
                         validation_steps = 10000)

print(".....................Probando el modelo...............")
toc()
target_dir = './modelo/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
classifier.save('./modelo/modelo1_dc.h5')
classifier.save_weights('./modelo/pesos1_dc.h5')


import numpy as np
from keras.preprocessing import image

print(".....................Resultado: 0 ...............")
tic()
test_image = image.load_img('BD_mov/TEST/0_IZ/izq0015.png', target_size = (Alto, Ancho))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
array = classifier.predict(test_image)
answer = array[0]
result= np.argmax(answer)
print(result)

test_image = image.load_img('BD_mov/TEST/0_IZ/izq0088.png', target_size = (Alto, Ancho))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
array = classifier.predict(test_image)
answer = array[0]
result= np.argmax(answer)
print(result)

print(".....................Resultado: 1 ...............")

test_image = image.load_img('BD_mov/TEST/1_SI/semiz0015.png', target_size = (Alto, Ancho))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
array = classifier.predict(test_image)
answer = array[0]
result= np.argmax(answer)
print(result)

test_image = image.load_img('BD_mov/TEST/1_SI/semiz0090.png', target_size = (Alto, Ancho))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
array = classifier.predict(test_image)
answer = array[0]
result= np.argmax(answer)
print(result)

print(".....................Resultado: 2 ...............")

test_image = image.load_img('BD_mov/TEST/2_AD/adelante0013.png', target_size = (Alto, Ancho))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
array = classifier.predict(test_image)
answer = array[0]
result= np.argmax(answer)
print(result)

test_image = image.load_img('BD_mov/TEST/2_AD/adelante0008.png', target_size = (Alto, Ancho))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
array = classifier.predict(test_image)
answer = array[0]
result= np.argmax(answer)
print(result)

print(".....................Resultado: 3 ...............")

test_image = image.load_img('BD_mov/TEST/3_SD/semder0013.png', target_size = (Alto, Ancho))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
array = classifier.predict(test_image)
answer = array[0]
result= np.argmax(answer)
print(result)

test_image = image.load_img('BD_mov/TEST/3_SD/semder0092.png', target_size = (Alto, Ancho))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
array = classifier.predict(test_image)
answer = array[0]
result= np.argmax(answer)
print(result)

print(".....................Resultado: 4 ...............")

test_image = image.load_img('BD_mov/TEST/4_DE/der0013.png', target_size = (Alto, Ancho))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
array = classifier.predict(test_image)
answer = array[0]
result= np.argmax(answer)
print(result)

test_image = image.load_img('BD_mov/TEST/4_DE/der0093.png', target_size = (Alto, Ancho))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
array = classifier.predict(test_image)
answer = array[0]
result= np.argmax(answer)
print(result)

toc()