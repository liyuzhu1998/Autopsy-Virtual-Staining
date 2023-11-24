# based on https://github.com/lopeneljxi/keras-unet-collection

from __future__ import absolute_import

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from .backbone_zoo import backbone_zoo, bach_norm_checker
from .layer_utils import *


def UNET_left(X, channel, kernel_size=3, stack_num=2, activation='ReLU',
              pool=True, batch_norm=False, name='left0'):
    '''
    The encoder block of U-net.

    UNET_left(X, channel, kernel_size=3, stack_num=2, activation='ReLU',
              pool=True, batch_norm=False, name='left0')

    Input
    ----------
        X: input tensor.
        channel: number of convolution filters.
        kernel_size: size of 2-d convolution kernels.
        stack_num: number of convolutional layers.
        activation: one of the `tensorflow.keras.layers` interface, e.g., 'ReLU'.
        pool: True or 'max' for MaxPooling2D.
              'ave' for AveragePooling2D.
              False for strided conv + batch norm + activation.
        batch_norm: True for batch normalization, False otherwise.
        name: prefix of the created keras layers.

    Output
    ----------
        X: output tensor.

    '''
    pool_size = 2

    X = encode_layer(X, channel, pool_size, pool, activation=activation,
                     batch_norm=batch_norm, name='{}_encode'.format(name))

    X = CONV_stack(X, channel, kernel_size, stack_num=stack_num, activation=activation,
                   batch_norm=batch_norm, name='{}_conv'.format(name))

    return X


def discriminator_base(input_tensor, filter_num, stack_num_down=2,
                       activation='ReLU', batch_norm=False, pool=True,
                       backbone=None, name='unet'):
    '''
    The base of discriminator CNN.

    discriminator_base(input_tensor, filter_num, stack_num_down=2,
                       activation='ReLU', batch_norm=False, pool=True,
                       backbone=None, name='unet')

    Input
    ----------
        input_tensor: the input tensor of the base, e.g., `keras.layers.Inpyt((None, None, 3))`.
        filter_num: a list that defines the number of filters for each \
                    down- and upsampling levels. e.g., `[64, 128, 256, 512]`.
                    The depth is expected as `len(filter_num)`.
        stack_num_down: number of convolutional layers per downsampling level/block.
        activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., 'ReLU'.
        batch_norm: True for batch normalization.
        pool: True or 'max' for MaxPooling2D.
              'ave' for AveragePooling2D.
              False for strided conv + batch norm + activation.
        name: prefix of the created keras model_G and its layers.


    Output
    ----------
        X: output tensor.

    '''

    X_skip = []

    X = input_tensor

    # stacked conv2d before downsampling
    X = CONV_stack(X, filter_num[0], stack_num=stack_num_down, activation=activation,
                   batch_norm=False, name='{}_down0'.format(name))
    X_skip.append(X)

    # downsampling blocks
    for i, f in enumerate(filter_num[1:]):
        X = UNET_left(X, f, stack_num=stack_num_down, activation=activation, pool=False,
                      batch_norm=False, name='{}_down{}'.format(name, i + 1))

    X = tf.reduce_mean(X, axis=(1, 2))
    ch = X.get_shape().as_list()[-1]
    X = dense_layer(X, units=ch, activation='LeakyReLU', name='dense_layer_1')
    X = dense_layer(X, units=1, activation=None, name='dense_layer_2')
    X = tf.nn.sigmoid(X)

    return X


def discriminator_2d(input_size, filter_num, stack_num_down=2,
                     activation='ReLU', batch_norm=False, pool=False,
                     backbone=None, name='unet'):
    '''
    U-net with an optional ImageNet-trained bakcbone.

    unet_2d(input_size, filter_num, n_labels, stack_num_down=2, stack_num_up=2,
            activation='ReLU', output_activation='Softmax', batch_norm=False, pool=True, unpool=True,
            backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='unet')

    ----------
    Ronneberger, O., Fischer, P. and Brox, T., 2015, October. U-net: Convolutional networks for biomedical image segmentation.
    In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). Springer, Cham.

    Input
    ----------
        input_size: the size/shape of network input, e.g., `(128, 128, 3)`.
        filter_num: a list that defines the number of filters for each \
                    down- and upsampling levels. e.g., `[64, 128, 256, 512]`.
                    The depth is expected as `len(filter_num)`.
        n_labels: number of output labels.
        stack_num_down: number of convolutional layers per downsampling level/block.
        stack_num_up: number of convolutional layers (after concatenation) per upsampling level/block.
        activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., 'ReLU'.
        output_activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interface or 'Sigmoid'.
                           Default option is 'Softmax'.
                           if None is received, then linear activation is applied.
        batch_norm: True for batch normalization.
        pool: True or 'max' for MaxPooling2D.
              'ave' for AveragePooling2D.
              False for strided conv + batch norm + activation.
        unpool: True or 'bilinear' for Upsampling2D with bilinear interpolation.
                'nearest' for Upsampling2D with nearest interpolation.
                False for Conv2DTranspose + batch norm + activation.
        name: prefix of the created keras model_G and its layers.

        ---------- (keywords of backbone options) ----------
        backbone_name: the bakcbone model_G name. Should be one of the `tensorflow.keras.applications` class.
                       None (default) means no backbone.
                       Currently supported backbones are:
                       (1) VGG16, VGG19
                       (2) ResNet50, ResNet101, ResNet152
                       (3) ResNet50V2, ResNet101V2, ResNet152V2
                       (4) DenseNet121, DenseNet169, DenseNet201
                       (5) EfficientNetB[0-7]
        weights: one of None (random initialization), 'imagenet' (pre-training on ImageNet),
                 or the path to the weights file to be loaded.
        freeze_backbone: True for a frozen backbone.
        freeze_batch_norm: False for not freezing batch normalization layers.

    Output
    ----------
        model_G: a keras model_G.

    '''
    activation_func = eval(activation)

    if backbone is not None:
        bach_norm_checker(backbone, batch_norm)

    IN = Input(input_size)

    # base
    X = discriminator_base(IN, filter_num, stack_num_down=stack_num_down,
                           activation=activation, batch_norm=batch_norm, pool=pool, name=name)

    # output layer
    OUT = X

    # functional API model_G
    model = Model(inputs=[IN, ], outputs=[OUT, ], name='{}_model'.format(name))

    return model
