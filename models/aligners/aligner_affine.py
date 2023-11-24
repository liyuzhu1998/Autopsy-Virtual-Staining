import numpy as np
# import keras.backend as K
from keras.models import Model
import keras.layers as KL
# from keras.layers import Layer
from keras.layers import Conv3D, Activation, Input, UpSampling3D, concatenate, AveragePooling2D
from keras.layers import LeakyReLU, Reshape, Lambda
from keras.initializers import RandomNormal
import keras.initializers
import tensorflow as tf
# from tensorflow.python.keras.layers import Lambda
# from keras.layers import DepthwiseConv2D, Conv2D
from .stn_affine import spatial_transformer_network
from .utils import affine_to_shift, batch_affine_to_shift
from .layers import SpatialTransformer


def unet_core_v3(vol_size, enc_nf, dec_nf, full_size=True, src=None, tgt=None, src_feats=3, tgt_feats=3):
    """
    unet architecture for voxelmorph models presented in the CVPR 2018 paper.
    You may need to modify this code (e.g., number of layers) to suit your project needs.
    :param vol_size: volume size. e.g. (256, 256, 256)
    :param enc_nf: list of encoder filters. right now it needs to be 1x4.
           e.g. [16,32,32,32]
    :param dec_nf: list of decoder filters. right now it must be 1x6 (like voxelmorph-1) or 1x7 (voxelmorph-2)
    :return: the keras model
    """
    ndims = len(vol_size)

    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims
    upsample_layer = getattr(KL, 'UpSampling%dD' % ndims)

    # inputs

    if src is None:
        src = Input(shape=[*vol_size, src_feats])
    if tgt is None:
        tgt = Input(shape=[*vol_size, tgt_feats])
    x_in = concatenate([src, tgt])

    print('x_in', x_in.shape)
    # down-sample path (encoder)
    x_enc = [x_in]
    for i in range(len(enc_nf)):
        x_enc.append(conv_block(x_enc[-1], enc_nf[i], 2))
    print('encoded', x_enc)
    # up-sample path (decoder)
    # x = conv_block(x_enc[-1], dec_nf[0])
    # x = upsample_layer()(x)
    # x = concatenate([x, x_enc[-2]])
    # x = conv_block(x, dec_nf[1])
    # x = upsample_layer()(x)
    # x = concatenate([x, x_enc[-3]])
    # x = conv_block(x, dec_nf[2])
    # x = upsample_layer()(x)
    # x = concatenate([x, x_enc[-4]])
    # x = conv_block(x, dec_nf[3])
    # x = conv_block(x, dec_nf[4])

    # only upsample to full dim if full_size
    # here we explore architectures where we essentially work with flow fields
    # that are 1/2 size
    # if full_size:
    #     x = upsample_layer()(x)
    #     x = concatenate([x, x_enc[0]])
    #     x = conv_block(x, dec_nf[5])
    # optional convolution at output resolution (used in voxelmorph-2)
    # if len(dec_nf) == 7:
    #     x = conv_block(x, dec_nf[6])

    # output hidden representation
    # return Model(inputs=[src, tgt], outputs=[x, x_enc[-1]])
    return Model(inputs=[src, tgt], outputs=x_enc[-1])



def aligner_unet_cvpr2018_v3(vol_size, enc_nf, dec_nf, full_size=True, indexing='ij'):
    """
    UNet with affine translation prediction stemming from the bottleneck representation.
    """
    ndims = len(vol_size)
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims

    # get the core model
    unet_model = unet_core_v3(vol_size, enc_nf, dec_nf, full_size=full_size)
    [moving, fixed] = unet_model.inputs

    bottleneck_repr = unet_model.output

    # get a shifting amount from hidden_repr
    flow = KL.Conv2D(64, kernel_size=3, padding='same', strides=(2, 2), name='flow_1',
                     kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(bottleneck_repr)     # (Batch, 8, 8, 64)
    flow = KL.AveragePooling2D(pool_size=(8, 8), name='flow_pooling_1')(flow)  # (Batch, 64)
    flow = KL.Flatten()(flow)
    # flow = KL.Dense(ndims)(flow)                                               # (Batch, 2)   (x_shift and y_shift)
    # flow = KL.Dense(4)(flow)                                               # (Batch, 4)   (tx, x_shift, ty, y_shift)
    flow = KL.Dense(2)(flow)                                               # (Batch, 4)   (tx, x_shift, ty, y_shift)

    # shifting & shear only for now
    flow_affine = tf.repeat(flow, [3, 3], axis=1)                              # (Batch, 6)
    # flow_affine = tf.repeat(flow, [2, 1, 2, 1], axis=1)                              # (Batch, 6)
    # clear out other entries
    flow_affine = flow_affine * tf.constant([0., 0., 1., 0., 0., 1.])
    # flow_affine = flow_affine * tf.constant([0., 1., 1., 1., 0., 1.])
    # fill in identity transform
    flow_affine = flow_affine + tf.constant([1., 0., 0., 0., 1., 0.])

    # transform to DVF and apply with SpatialTransformer
    # flow_affine = batch_affine_to_shift(flow_affine, vol_size)
    # moving_transformed = SpatialTransformer(interp_method='linear', indexing=indexing)([moving, flow_affine])
    moving_transformed = spatial_transformer_network(moving, flow_affine)

    model = Model(inputs=[moving, fixed], outputs=[moving_transformed, flow_affine])
    return model


def unet_core_v4(vol_size, enc_nf, dec_nf, full_size=True, src=None, tgt=None, src_feats=3, tgt_feats=3):
    """
    unet architecture for voxelmorph models presented in the CVPR 2018 paper.
    You may need to modify this code (e.g., number of layers) to suit your project needs.
    :param vol_size: volume size. e.g. (256, 256, 256)
    :param enc_nf: list of encoder filters. right now it needs to be 1x4.
           e.g. [16,32,32,32]
    :param dec_nf: list of decoder filters. right now it must be 1x6 (like voxelmorph-1) or 1x7 (voxelmorph-2)
    :return: the keras model
    """
    ndims = len(vol_size)

    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims
    upsample_layer = getattr(KL, 'UpSampling%dD' % ndims)

    # inputs

    if src is None:
        src = Input(shape=[*vol_size, src_feats])
    if tgt is None:
        tgt = Input(shape=[*vol_size, tgt_feats])
    x_in = concatenate([src, tgt])

    print('x_in', x_in.shape)
    # down-sample path (encoder)
    x_enc = [x_in]
    for i in range(len(enc_nf)):
        x_enc.append(conv_block_v2(x_enc[-1], enc_nf[i], 2))
    print('encoded', x_enc)
    # up-sample path (decoder)
    # x = conv_block_v2(x_enc[-1], dec_nf[0])
    # x = upsample_layer()(x)
    # x = concatenate([x, x_enc[-2]])
    # x = conv_block_v2(x, dec_nf[1])
    # x = upsample_layer()(x)
    # x = concatenate([x, x_enc[-3]])
    # x = conv_block_v2(x, dec_nf[2])
    # x = upsample_layer()(x)
    # x = concatenate([x, x_enc[-4]])
    # x = conv_block_v2(x, dec_nf[3])
    # x = conv_block_v2(x, dec_nf[4])

    # only upsample to full dim if full_size
    # here we explore architectures where we essentially work with flow fields
    # that are 1/2 size
    # if full_size:
    #     x = upsample_layer()(x)
    #     x = concatenate([x, x_enc[0]])
    #     x = conv_block_v2(x, dec_nf[5])
    # optional convolution at output resolution (used in voxelmorph-2)
    # if len(dec_nf) == 7:
    #     x = conv_block_v2(x, dec_nf[6])

    # output hidden representation
    # return Model(inputs=[src, tgt], outputs=[x, x_enc[-1]])
    return Model(inputs=[src, tgt], outputs=x_enc[-1])


def unet_core_v4_residual(vol_size, enc_nf, dec_nf, full_size=True, src=None, tgt=None, src_feats=3, tgt_feats=3):
    """
    unet architecture for voxelmorph models presented in the CVPR 2018 paper.
    You may need to modify this code (e.g., number of layers) to suit your project needs.
    :param vol_size: volume size. e.g. (256, 256, 256)
    :param enc_nf: list of encoder filters. right now it needs to be 1x4.
           e.g. [16,32,32,32]
    :param dec_nf: list of decoder filters. right now it must be 1x6 (like voxelmorph-1) or 1x7 (voxelmorph-2)
    :return: the keras model
    """
    ndims = len(vol_size)

    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims
    upsample_layer = getattr(KL, 'UpSampling%dD' % ndims)

    # inputs

    if src is None:
        src = Input(shape=[*vol_size, src_feats])
    if tgt is None:
        tgt = Input(shape=[*vol_size, tgt_feats])
    x_in = concatenate([src, tgt])

    print('x_in', x_in.shape)
    # down-sample path (encoder)
    x_enc = [x_in]
    for i in range(len(enc_nf)):
        x_enc.append(conv_block_v2_residual(x_enc[-1], enc_nf[i]))
    print('encoded', x_enc)
    # up-sample path (decoder)
    # x = conv_block_v2(x_enc[-1], dec_nf[0])
    # x = upsample_layer()(x)
    # x = concatenate([x, x_enc[-2]])
    # x = conv_block_v2_residual(x, dec_nf[1])
    # x = upsample_layer()(x)
    # x = concatenate([x, x_enc[-3]])
    # x = conv_block_v2_residual(x, dec_nf[2])
    # x = upsample_layer()(x)
    # x = concatenate([x, x_enc[-4]])
    # x = conv_block_v2_residual(x, dec_nf[3])
    # x = conv_block_v2_residual(x, dec_nf[4])

    # only upsample to full dim if full_size
    # here we explore architectures where we essentially work with flow fields
    # that are 1/2 size
    # if full_size:
    #     x = upsample_layer()(x)
    #     x = concatenate([x, x_enc[0]])
    #     x = conv_block_v2_residual(x, dec_nf[5])
    # optional convolution at output resolution (used in voxelmorph-2)
    # if len(dec_nf) == 7:
    #     x = conv_block_v2_residual(x, dec_nf[6])

    # output hidden representation
    # return Model(inputs=[src, tgt], outputs=[x, x_enc[-1]])
    return Model(inputs=[src, tgt], outputs=x_enc[-1])



def aligner_unet_cvpr2018_v4(vol_size, enc_nf, dec_nf, full_size=True, indexing='ij',
                             loss_mask=False, loss_mask_from_prev_cascade=False):
    """
    UNet with affine translation prediction stemming from the bottleneck representation.
    """
    img_size = vol_size[0]
    n_levels = len(enc_nf)
    pooling_window_size = img_size // 2**(n_levels+1)

    ndims = len(vol_size)
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims

    # get the core model
    unet_model = unet_core_v4_residual(vol_size, enc_nf, dec_nf, full_size=full_size)
    [moving, fixed] = unet_model.inputs
    bottleneck_repr = unet_model.output

    if loss_mask:
        if loss_mask_from_prev_cascade:
            # use cascaded mask from previous aligner
            moving =Input(shape=[*vol_size, 3+1])
            tgt = moving[:, :, :, :-1]
            train_loss_mask = moving[:, :, :, -1:]
            bottleneck_repr = unet_model(tgt, fixed)
        else:
            # initialize a new mask
            train_loss_mask = tf.ones([moving.get_shape().as_list()[0], vol_size[0] - 4, vol_size[1] - 4, 1],
                                      dtype=tf.float32)
            paddings = tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]])
            train_loss_mask = tf.pad(train_loss_mask, paddings, mode="CONSTANT")

    # get a shifting amount from hidden_repr
    flow = KL.Conv2D(64, kernel_size=3, padding='same', strides=(2, 2), name='flow_1',
                     kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(bottleneck_repr)     # (Batch, pooling_window_size, pooling_window_size, 64)
    flow = KL.AveragePooling2D(pool_size=(pooling_window_size, pooling_window_size), name='flow_pooling_1')(flow)  # (Batch, 64)
    flow = KL.Flatten()(flow)
    # flow = KL.Dense(ndims)(flow)                                               # (Batch, 2)   (x_shift and y_shift)
    flow = KL.Dense(4)(flow)                                               # (Batch, 4)   (tx, x_shift, ty, y_shift)
    # flow = KL.Dense(2)(flow)                                               # (Batch, 4)   (tx, x_shift, ty, y_shift)

    # shifting & shear only for now
    # flow_affine = tf.repeat(flow, [3, 3], axis=1)                              # (Batch, 6)
    flow_affine = tf.repeat(flow, [2, 1, 2, 1], axis=1)                              # (Batch, 6)
    # clear out other entries
    # flow_affine = flow_affine * tf.constant([0., 0., 1., 0., 0., 1.])
    flow_affine = flow_affine * tf.constant([0., 1., 1., 1., 0., 1.])
    # fill in identity transformqw
    flow_affine = flow_affine + tf.constant([1., 0., 0., 0., 1., 0.])

    # append a mask in the channel dimension of moving for G loss masking
    if loss_mask:
        moving_for_warp = tf.concat([moving, train_loss_mask], axis=-1)
    else:
        moving_for_warp = moving

    # transform to DVF and apply with SpatialTransformer
    # flow_affine = batch_affine_to_shift(flow_affine, vol_size)
    # moving_transformed = SpatialTransformer(interp_method='linear', indexing=indexing)([moving, flow_affine])
    moving_transformed = spatial_transformer_network(moving_for_warp, flow_affine)

    model = Model(inputs=[moving, fixed], outputs=[moving_transformed, flow_affine])
    return model



########################################################
# Helper functions
########################################################

def conv_block(x_in, nf, strides=1):
    """
    specific convolution module including convolution followed by leakyrelu
    """
    ndims = len(x_in.get_shape()) - 2
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims

    Conv = getattr(KL, 'Conv%dD' % 2)
    x_out = Conv(nf, kernel_size=3, padding='same',
                 kernel_initializer='he_normal', strides=strides)(x_in)
    x_out = LeakyReLU(0.2)(x_out)
    return x_out


def conv_block_v2(x_in, nf, strides=1):
    """
    specific convolution module including convolution followed by leakyrelu
    """
    ndims = len(x_in.get_shape()) - 2
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims

    Conv1 = getattr(KL, 'Conv%dD' % 2)
    x_mid = Conv1(nf, kernel_size=3, padding='same',
                 kernel_initializer='he_normal', strides=strides)(x_in)
    x_mid = LeakyReLU(0.2)(x_mid)

    Conv2 = getattr(KL, 'Conv%dD' % 2)
    x_out = Conv2(nf, kernel_size=3, padding='same',
                 kernel_initializer='he_normal', strides=1)(x_mid)
    x_out = LeakyReLU(0.2)(x_out)
    return x_out


def conv_block_v2_residual(x_in, nf):
    """
    specific convolution module including convolution followed by leakyrelu
    Two 3x3 stride 1 CONV, followed by one 2x2 average pooling.
    """
    ndims = len(x_in.get_shape()) - 2
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims

    Conv1 = getattr(KL, 'Conv%dD' % 2)
    x_mid = Conv1(nf, kernel_size=3, padding='same',
                 kernel_initializer='he_normal', strides=1)(x_in)
    x_mid = LeakyReLU(0.2)(x_mid)

    Conv2 = getattr(KL, 'Conv%dD' % 2)
    x_out = Conv2(nf, kernel_size=3, padding='same',
                 kernel_initializer='he_normal', strides=1)(x_mid)

    x_in_padded = tf.pad(x_in, [[0, 0], [0, 0], [0, 0], [0, nf - x_in.get_shape().as_list()[-1]]], 'CONSTANT')
    x_out = x_out + x_in_padded                # residual connection
    x_out = LeakyReLU(0.2)(x_out)

    x_out = AveragePooling2D(pool_size=(2, 2))(x_out)

    return x_out
