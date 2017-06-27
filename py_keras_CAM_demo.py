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

#--Replace the path for keras with local path--------
#--- Local path ---
sys.path.insert(0,"/media/manish/Data/keras/keras")
sys.path.insert(0,"/media/manish/Data/keras/keras/keras")

#--- Server path ---
#sys.path.insert(0,"/data/Manish/MedDCH/keras")
#sys.path.insert(0,"/dedia/Manish/MedDCH/keras/keras")
#-----------------------------------------------------

from keras import backend as K
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import matplotlib.pylab as plt
import h5py
import numpy as np

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

# code modified from https://github.com/tdeboissiere/VGG16CAM-keras/blob/master/VGGCAM-keras.py
def get_classmap(model, X, nb_classes, batch_size, num_input_channels=512):
    inc = model.layers[0].input
    conv5 = model.layers[-6].output
    conv5_resized = K.tf.image.resize_bilinear( conv5, [32, 32] )
    WT =K.transpose(model.layers[-2].W)
    conv5_resized = K.reshape(conv5_resized, (batch_size, num_input_channels, 32 * 32))
    classmap = K.dot(WT, conv5_resized)
    classmap = K.reshape(classmap, (batch_size, nb_classes, 32, 32))
    get_cmap = K.function([inc], [classmap])
    return get_cmap([X])[0]

def getModel():
    with K.tf.device('/cpu:0'):
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
        return model                 

model = getModel()
model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])


model.load_weights(weights_path, by_name=True)
print('Model loaded.')

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
batch_size = 1

im =  X_test[20]
imcp = im.copy()
im = im.reshape(-1, 32,32, 3)
output_val = model.predict(im, batch_size=batch_size, verbose=0)
label = output_val.argmax( axis=1 )[0]

classmap = get_classmap(model,
                         im,
                         nb_classes,
                         batch_size)

classmap = np.array(map(lambda x: ((x-x.min())/(x.max()-x.min())), classmap))

plt.imshow(imcp)
plt.imshow(classmap[0, label, :, :],
           cmap="jet",
           alpha=0.3,
           interpolation='nearest')
plt.show()
