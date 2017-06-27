'''Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn.py

It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import sys
import os
# import numpy as np
# os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu2,floatX=float32,lib.cnmem=0.8,exception_verbosity=high"

#--- Local path ---
#sys.path.insert(0,"/media/manish/Data/keras/keras")
#sys.path.insert(0,"/media/manish/Data/keras/keras/keras")

#--- Server path ---
sys.path.insert(0,"/data/Manish/MedDCH/keras")
sys.path.insert(0,"/dedia/Manish/MedDCH/keras/keras")
from keras import backend as K
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

batch_size = 32
nb_classes = 10
nb_epoch = 200
data_augmentation = True

# input image dimensions
img_rows, img_cols = 32, 32
# The CIFAR10 images are RGB.
img_channels = 3
weights_path = './cifar10/cifar10.h5' 

# # The data, shuffled and split between train and test sets:
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

#--- functional form -------
if K.image_dim_ordering() == 'th':
    print("image ordering = th")
    input_shape = (img_channels, img_rows, img_cols)
    _axis=1
else:
    input_shape = (img_rows, img_cols, img_channels)
    print("image ordering = tf")

def getModel():
    with K.tf.device('/gpu:1'):
	config = K.tf.ConfigProto(allow_soft_placement=False, log_device_placement=True)
	config.gpu_options.allow_growth = True
        K.set_session(K.tf.Session(config = config))
        input_img = Input(shape=input_shape)
        conv_1 = Convolution2D(32, 3, 3, border_mode='same')(input_img)
        relu_1= Activation('relu')(conv_1)
        conv_2 = Convolution2D(32, 3, 3)(relu_1)
        relu_2 = Activation('relu')(conv_2)
        maxpool_1 = MaxPooling2D(pool_size=(2,2))(relu_2)
        dropout_1 = Dropout(0.25)(maxpool_1)

        conv_3 = Convolution2D(64, 3, 3, border_mode='same')(maxpool_1)
        relu_3 = Activation('relu')(conv_3)
        conv_4 = Convolution2D(64, 3, 3)(relu_3)
        relu_4 = Activation('relu')(conv_4)

        # Add another conv layer with ReLU + GAP
        conv_5 = Convolution2D(512, 3, 3, border_mode='same')(relu_4)
        gap = AveragePooling2D((13, 13)) (conv_5)
        dropout_2 = Dropout(0.5)(gap)
        flat = Flatten()(dropout_2)

        softmax_class = Dense(nb_classes,name="class")(flat)
        output = Activation('softmax', name="output")(softmax_class)
        #--------------------------------------

        model = Model(input=input_img, output=output)

        # Let's train the model using RMSprop
        model.compile(loss='categorical_crossentropy',
                    optimizer='rmsprop',

                    metrics=['accuracy'])        
        return model                 

model = getModel()
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

saveBestModel = ModelCheckpoint(weights_path, monitor='val_loss', 
                                    verbose=1, save_best_only=True, save_weights_only=True,period=5)

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test),
              shuffle=True,
              callbacks=[saveBestModel])
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(X_train)
    
    

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(X_train, Y_train,
                        batch_size=batch_size),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch,
                        validation_data=(X_test, Y_test),
                        verbose=1,
                        callbacks=[saveBestModel])
