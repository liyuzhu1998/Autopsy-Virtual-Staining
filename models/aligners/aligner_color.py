import numpy as np
# import keras.backend as K
from keras.models import Model
import keras.layers as KL
from keras.layers import Layer
from keras.layers import Conv3D, Activation, Input, UpSampling3D, concatenate, AveragePooling2D
from keras.layers import LeakyReLU, Reshape, Lambda
from keras.initializers import RandomNormal
import keras.initializers
import tensorflow as tf
# import tensorflow_addons as tfa
# from tensorflow.python.keras.layers import Lambda
# from keras.layers import DepthwiseConv2D, Conv2D
import color_ops


def rgb2lms(rgb_img):
    out_shape = rgb_img.get_shape().as_list()

    RGB2LMS = tf.constant(np.array([[0.3811, 0.5783, 0.0402],
                                    [0.1967, 0.7244, 0.0782],
                                    [0.0241, 0.1288, 0.8444]]), dtype='float32')

    rgb_img = tf.expand_dims(rgb_img, -1)
    for _ in range(len(rgb_img.get_shape().as_list()) - 2):
        RGB2LMS = tf.expand_dims(RGB2LMS, 0)

    return tf.reshape(tf.matmul(RGB2LMS, rgb_img), out_shape)


def lms2lab(lms_img):
    out_shape = lms_img.get_shape().as_list()

    T1 = tf.constant(np.array([[1, 1, 1],
                               [1, 1, -2],
                               [1, -1, 0]]), dtype='float32')

    T2 = tf.constant(np.array([[1/np.sqrt(3), 0, 0],
                               [0, 1/np.sqrt(6), 0],
                               [0, 0, 1/np.sqrt(2)]]), dtype='float32')

    lms_img = tf.expand_dims(lms_img, -1)
    for _ in range(len(lms_img.get_shape().as_list()) - 2):
        T1 = tf.expand_dims(T1, 0)
        T2 = tf.expand_dims(T2, 0)

    return tf.reshape(tf.matmul(T2, tf.matmul(T1, lms_img)), out_shape)


def lab2lms(lab_img):
    out_shape = lab_img.get_shape().as_list()

    T1 = tf.constant(np.array([[np.sqrt(3)/3, 0, 0],
                               [0, np.sqrt(6)/6, 0],
                               [0, 0, np.sqrt(2)/2]]), dtype='float32')

    T2 = tf.constant(np.array([[1, 1, 1],
                               [1, 1, -1],
                               [1, -2, 0]]), dtype='float32')

    lab_img = tf.expand_dims(lab_img, -1)
    for _ in range(len(lab_img.get_shape().as_list()) - 2):
        T1 = tf.expand_dims(T1, 0)
        T2 = tf.expand_dims(T2, 0)

    return tf.reshape(tf.matmul(T2, tf.matmul(T1, lab_img)), out_shape)


def lms2rgb(lms_img):
    out_shape = lms_img.get_shape().as_list()

    LMS2RGB = tf.constant(np.array([[4.4679, -3.5873, 0.1193],
                                    [-1.2186, 2.3809, -0.1624],
                                    [0.0497, -0.2439, 1.2045]]), dtype='float32')

    lms_img = tf.expand_dims(lms_img, -1)
    for _ in range(len(lms_img.get_shape().as_list()) - 2):
        LMS2RGB = tf.expand_dims(LMS2RGB, 0)

    return tf.reshape(tf.matmul(LMS2RGB, lms_img), out_shape)


def rgb2lab_tf(img_rgb):
        img_rgb = tf.clip_by_value(img_rgb, 0.0, 255.0)
        img_lms = rgb2lms(img_rgb)
        img_lms = tf.clip_by_value(img_lms, 1e-10, 1e10)
        img_lms = tf.math.log(img_lms)
        img_lab = lms2lab(img_lms)

        '''
        if tf.reduce_any(tf.math.is_nan(img_lms), axis=(0, 1, 2, 3)):
            for i, sub_img in enumerate(img_lms):
                if tf.reduce_any(tf.math.is_nan(sub_img), axis=(0, 1, 2)):
                    print(sub_img)
                    print(tf.reduce_max(img_rgb[i]), tf.reduce_min(img_rgb[i]))
                    # print(img_rgb)
                    # print(tf.reduce_any(tf.math.is_nan(img_lms), axis=3))
                    exit()
        '''

        return img_lab


def lab2rgb_tf(img_lab):
    img_lms = lab2lms(img_lab)
    img_lms = tf.math.exp(img_lms)
    img_rgb = lms2rgb(img_lms)
    return img_rgb


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



def color_aligner_unet_cvpr2018_v4(vol_size, enc_nf, dec_nf, full_size=True, indexing='ij', fix_hsv_value=None,
                                   hsv_value_regularizer=None):
    """
    UNet with color adjustment prediction stemming from the bottleneck representation.
    """
    img_size = vol_size[0]
    n_levels = len(enc_nf)
    pooling_window_size = img_size // 2**(n_levels+1)

    ndims = len(vol_size)
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims

    if fix_hsv_value is not None:
        assert type(fix_hsv_value) == float

    # get the core model
    unet_model = unet_core_v4_residual(vol_size, enc_nf, dec_nf, full_size=full_size)
    [moving, fixed] = unet_model.inputs
    bottleneck_repr = unet_model.output

    # get a shifting amount from hidden_repr
    flow = KL.Conv2D(64, kernel_size=3, padding='same', strides=(2, 2), name='flow_1',
                     kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(bottleneck_repr)     # (Batch, pooling_window_size, pooling_window_size, 64)
    flow = KL.AveragePooling2D(pool_size=(pooling_window_size, pooling_window_size), name='flow_pooling_1')(flow)  # (Batch, 64)
    flow = KL.Flatten()(flow)
    if fix_hsv_value is None:
        flow = KL.Dense(3, activity_regularizer=hsv_value_regularizer)(flow)    # (Batch, 3)   (hue, saturation, value)
    else:
        flow = KL.Dense(2, activity_regularizer=hsv_value_regularizer)(flow)    # (Batch, 2)   (hue, saturation)

    hue_deltas = tf.clip_by_value(flow[:, 0:1], -1, 1)
    saturation_factors = flow[:, 1:2]
    if fix_hsv_value is None:
        scale_values = flow[:, 2:3]
    else:
        scale_values = tf.ones_like(saturation_factors, dtype=saturation_factors.dtype) * fix_hsv_value

    # print(moving.shape, hue_deltas.shape, saturation_factors.shape)

    class MapLayer(Layer):
        def call(self, moving, hue_delta, saturation_factor, scale_values):
            def adjust_hsv_in_yiq(inp):
                img, hue_delta, saturation_factor, scale_values = inp
                delta_hue = tf.squeeze(hue_delta)
                scale_saturation = tf.squeeze(saturation_factor)
                scale_value = tf.squeeze(scale_values)

                assert img.dtype in [tf.float16, tf.float32, tf.float64]

                if img.shape.rank is not None and img.shape.rank < 3:
                    raise ValueError("input must be at least 3-D.")
                if img.shape[-1] is not None and img.shape[-1] != 3:
                    raise ValueError(
                        "input must have 3 channels but instead has {}.".format(img.shape[-1])
                    )

                # Construct hsv linear transformation matrix in YIQ space.
                # https://beesbuzz.biz/code/hsv_color_transforms.php
                yiq = tf.constant([[0.299, 0.596, 0.211],
                                   [0.587, -0.274, -0.523],
                                   [0.114, -0.322, 0.312]],
                                  dtype=img.dtype)
                yiq_inverse = tf.constant(
                    [[1.0, 1.0, 1.0],
                     [0.95617069, -0.2726886, -1.103744],
                     [0.62143257, -0.64681324, 1.70062309]],
                    dtype=img.dtype)

                vsu = scale_value * scale_saturation * tf.math.cos(delta_hue)
                vsw = scale_value * scale_saturation * tf.math.sin(delta_hue)
                hsv_transform = tf.convert_to_tensor([[scale_value, 0, 0],
                                                      [0, vsu, vsw],
                                                      [0, -vsw, vsu]],
                                                     dtype=img.dtype)
                transform_matrix = yiq @ hsv_transform @ yiq_inverse

                img = img @ transform_matrix

                return (img, hue_delta, saturation_factor, scale_values)

            ret = tf.map_fn(adjust_hsv_in_yiq, (moving, hue_delta, saturation_factor, scale_values))

            return ret[0]
            # return tf.stack(res, axis=0)

    # moving_hue_adjusted = tf.map_fn(tf.image.adjust_hue, (moving, hue_deltas))
    # moving_all_adjusted = tf.map_fn(tf.image.adjust_saturation, (moving_hue_adjusted, saturation_factors))
    moving_adjusted = MapLayer()(moving,
                                 tf.expand_dims(tf.expand_dims(hue_deltas, axis=-1), axis=-1),
                                 tf.expand_dims(tf.expand_dims(saturation_factors, axis=-1), axis=-1),
                                 tf.expand_dims(tf.expand_dims(scale_values, axis=-1), axis=-1))
    # moving_adjusted = tf.map_fn(color_ops.adjust_hsv_in_yiq, (moving, hue_deltas, saturation_factors, dummy_scale_values))
    # moving_adjusted = tf.map_fn(color_ops.adjust_hsv_in_yiq, (moving,
    #                             tf.expand_dims(tf.expand_dims(hue_deltas, axis=0), axis=0),
    #                             tf.expand_dims(tf.expand_dims(saturation_factors, axis=0), axis=0),
    #                             tf.expand_dims(tf.expand_dims(dummy_scale_values, axis=0), axis=0)))
    # temporary hack to make unstack work
    # moving = tf.ensure_shape(moving, [16] + moving.get_shape().as_list()[1:])
    # hue_deltas = tf.ensure_shape(hue_deltas, [16] + hue_deltas.get_shape().as_list()[1:])
    # saturation_factors = tf.ensure_shape(saturation_factors, [16] + saturation_factors.get_shape().as_list()[1:])
    # moving_adjusted = tf.stack([tfa.image.adjust_hsv_in_yiq(a, b, c) for a, b, c in
    #                             zip(tf.unstack(moving), tf.unstack(hue_deltas), tf.unstack(saturation_factors))])
    # moving_adjusted = tfa.image.adjust_hsv_in_yiq(moving, hue_deltas, saturation_factors)

    model = Model(inputs=[moving, fixed], outputs=[moving_adjusted, flow])
    return model


def color_aligner_lab_unet_cvpr2018_v4(vol_size, enc_nf, dec_nf, full_size=True, indexing='ij'):
    """
    UNet with color adjustment prediction stemming from the bottleneck representation.
    Input should be converted to lab color space
    """
    img_size = vol_size[0]
    n_levels = len(enc_nf)
    pooling_window_size = img_size // 2 ** (n_levels + 1)

    ndims = len(vol_size)
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims

    # get the core model
    unet_model = unet_core_v4_residual(vol_size, enc_nf, dec_nf, full_size=full_size)
    [moving_lab, fixed_lab] = unet_model.inputs
    bottleneck_repr = unet_model.output

    # get a shifting amount from hidden_repr
    flow = KL.Conv2D(64, kernel_size=3, padding='same', strides=(2, 2), name='flow_1',
                     kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(
        bottleneck_repr)  # (Batch, pooling_window_size, pooling_window_size, 64)
    flow = KL.AveragePooling2D(pool_size=(pooling_window_size, pooling_window_size), name='flow_pooling_1')(
        flow)  # (Batch, 64)
    flow = KL.Flatten()(flow)
    flow = KL.Dense(6)(flow)    # (Batch, 6)   (l_mean, a_mean, b_mean, l_std, a_std, b_std)

    pred_target_mean = tf.expand_dims(tf.expand_dims(flow[:, 0:3], 1), 1)
    pred_target_std = tf.expand_dims(tf.expand_dims(flow[:, 3:6], 1), 1)
    source_mean, source_std = tf.math.reduce_mean(moving_lab, axis=(1, 2), keepdims=True), \
                              tf.math.reduce_std(moving_lab, axis=(1, 2), keepdims=True)

    moving_lab_transformed = (moving_lab - source_mean) * (pred_target_std / source_std) + pred_target_mean

    model = Model(inputs=[moving_lab, fixed_lab], outputs=[moving_lab_transformed, flow])
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
