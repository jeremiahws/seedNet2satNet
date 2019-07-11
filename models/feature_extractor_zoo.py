

'''seedNet2satNet/models/feature_extractor_zoo.py

Library of feature extractors for sliding-window CNNs.
'''


from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPool2D, Activation, Flatten, Dense, Dropout
from tensorflow.keras.models import Model


def VGG_like(input_shape, output_neurons, softmax_output=False, dropout=True):
    '''VGG16-like feature extractor.

    :param input_shape: tuple defining the input shape
    :param output_neurons: integer number of output neurons
    :param softmax_output: whether to apply a softmax to the output
    :param dropout: whether to apply dropout before the last dense layer
    :return: a keras model
    '''
    inputs = Input(shape=input_shape)
    outputs = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation('relu')(outputs)
    outputs = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation('relu')(outputs)
    outputs = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(outputs)
    outputs = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation('relu')(outputs)
    outputs = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation('relu')(outputs)
    outputs = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(outputs)
    outputs = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation('relu')(outputs)
    outputs = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation('relu')(outputs)
    outputs = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation('relu')(outputs)
    outputs = Flatten()(outputs)
    outputs = Dense(1024)(outputs)
    outputs = Activation('relu')(outputs)

    if dropout:
        outputs = Dropout(0.5)(outputs)

    outputs = Dense(output_neurons)(outputs)

    if softmax_output:
        outputs = Activation('softmax')(outputs)
    else:
        outputs = Activation('relu')(outputs)

    return Model(inputs=inputs, outputs=outputs)
