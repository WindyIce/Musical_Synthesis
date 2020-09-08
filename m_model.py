import numpy as np
import os

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras import backend as keras
from keras.activations import *
from keras.layers import LeakyReLU
from keras.losses import *
from keras import losses

lReLU_alpha_G=0.05
lReLU_alpha_D=0.05

learning_rate=1e-5


def generator(pretrained_weights=None, input_size=(1,1,317)):
    inputs=Input(input_size)
    conv1 = Conv2D(256, (2,16), padding='same', kernel_initializer='he_normal')(inputs)
    conv1=LeakyReLU(alpha=lReLU_alpha_G)(conv1)
    conv1 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = LeakyReLU(alpha=lReLU_alpha_G)(conv1)

    up1=UpSampling2D(size=(2,2))(conv1)

    conv2 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(up1)
    conv2 = LeakyReLU(alpha=lReLU_alpha_G)(conv2)
    conv2 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = LeakyReLU(alpha=lReLU_alpha_G)(conv2)

    up2 = UpSampling2D(size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(up2)
    conv3 = LeakyReLU(alpha=lReLU_alpha_G)(conv3)
    conv3 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = LeakyReLU(alpha=lReLU_alpha_G)(conv3)

    up3 = UpSampling2D(size=(2, 2))(conv3)

    conv4 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(up3)
    conv4 = LeakyReLU(alpha=lReLU_alpha_G)(conv4)
    conv4 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = LeakyReLU(alpha=lReLU_alpha_G)(conv4)

    up4 = UpSampling2D(size=(2, 2))(conv4)

    conv5 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(up4)
    conv5 = LeakyReLU(alpha=lReLU_alpha_G)(conv5)
    conv5 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = LeakyReLU(alpha=lReLU_alpha_G)(conv5)

    up5 = UpSampling2D(size=(2, 2))(conv5)

    conv6 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(up5)
    conv6 = LeakyReLU(alpha=lReLU_alpha_G)(conv6)
    conv6 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = LeakyReLU(alpha=lReLU_alpha_G)(conv6)

    up6 = UpSampling2D(size=(2, 2))(conv6)

    conv7 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(up6)
    conv7 = LeakyReLU(alpha=lReLU_alpha_G)(conv7)
    conv7 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = LeakyReLU(alpha=lReLU_alpha_G)(conv7)

    out = Conv2D(2, (1, 1), activation='tanh')(conv7)

    model=Model(inputs=inputs,outputs=out)

    model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy' )


def descriminator(pretrained_weights=None, input_size=(128,1024,2)):
    inputs=Input(input_size)

    conv1 = Conv2D(32, 1, padding='same', kernel_initializer='he_normal')(inputs)

    conv1 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = LeakyReLU(alpha=lReLU_alpha_D)(conv1)
    conv1 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = LeakyReLU(alpha=lReLU_alpha_D)(conv1)

    down1 = MaxPooling2D(pool_size=(2,2))(conv1)

    conv2 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(down1)
    conv2 = LeakyReLU(alpha=lReLU_alpha_D)(conv2)
    conv2 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = LeakyReLU(alpha=lReLU_alpha_D)(conv2)

    down2 = MaxPooling2D(pool_size=(2,2))(conv2)

    conv3 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(down2)
    conv3 = LeakyReLU(alpha=lReLU_alpha_D)(conv3)
    conv3 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = LeakyReLU(alpha=lReLU_alpha_D)(conv3)

    down3 = MaxPooling2D(pool_size=(2,2))(conv3)

    conv4 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(down3)
    conv4 = LeakyReLU(alpha=lReLU_alpha_D)(conv4)
    conv4 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = LeakyReLU(alpha=lReLU_alpha_D)(conv4)

    down4 = MaxPooling2D(pool_size=(2,2))(conv4)

    conv5 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(down4)
    conv5 = LeakyReLU(alpha=lReLU_alpha_D)(conv5)
    conv5 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = LeakyReLU(alpha=lReLU_alpha_D)(conv5)

    down5 = MaxPooling2D(pool_size=(2,2))(conv5)

    conv6 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(down5)
    conv6 = LeakyReLU(alpha=lReLU_alpha_D)(conv6)
    conv6 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = LeakyReLU(alpha=lReLU_alpha_D)(conv6)

    down6 = MaxPooling2D(pool_size=(2,2))(conv6)

    conv7 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(down6)
    conv7 = LeakyReLU(alpha=lReLU_alpha_D)(conv7)
    conv7 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = LeakyReLU(alpha=lReLU_alpha_D)(conv7)

    flatten1=Flatten()(conv7)

    out1=Dense(units=61, kernel_initializer="he_normal",
                      activation="softmax")(flatten1)

    out2 = Dense(units=1, kernel_initializer="he_normal",
                 activation="softmax")(flatten1)

    model = Model(inputs=inputs, outputs=[out1,out2])

    model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy')

    return model

import tensorflow as tf
import tensorflow_gan as tfgan
# tfgan = tf.contrib.gan
from model import Model

def gn_de(path):
    return Model.load_from_path(path)






