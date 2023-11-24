from __future__ import absolute_import

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from color_ops import rgb2hsl_tf, rgb2yiq_tf


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None, eps=1e-5):
        self.win = win
        self.eps = eps

    def ncc(self, Ii, Ji):
        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(Ii.get_shape().as_list()) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        if self.win is None:
            self.win = [9] * ndims
        elif not isinstance(self.win, list):  # user specified a single number not a list
            self.win = [self.win] * ndims

        # get convolution function
        conv_fn = getattr(tf.nn, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        # compute filters
        in_ch = Ji.get_shape().as_list()[-1]
        sum_filt = tf.ones([*self.win, in_ch, 1])
        strides = 1
        if ndims > 1:
            strides = [1] * (ndims + 2)

        # compute local sums via convolution
        padding = 'SAME'
        I_sum = conv_fn(Ii, sum_filt, strides, padding)
        J_sum = conv_fn(Ji, sum_filt, strides, padding)
        I2_sum = conv_fn(I2, sum_filt, strides, padding)
        J2_sum = conv_fn(J2, sum_filt, strides, padding)
        IJ_sum = conv_fn(IJ, sum_filt, strides, padding)

        # compute cross correlation
        win_size = np.prod(self.win) * in_ch
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        # TODO: simplify this
        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        cross = tf.maximum(cross, self.eps)
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        I_var = tf.maximum(I_var, self.eps)
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
        J_var = tf.maximum(J_var, self.eps)

        # cc = (cross * cross) / (I_var * J_var)
        cc = (cross / I_var) * (cross / J_var)

        # return mean cc for each entry in batch
        return tf.reduce_mean(K.batch_flatten(cc), axis=-1)

    def loss(self, y_true, y_pred):
        return 1 - self.ncc(y_true, y_pred)


class Grad:
    """
    N-D gradient loss.
    loss_mult can be used to scale the loss value - this is recommended if
    the gradient is computed on a downsampled vector field (where loss_mult
    is equal to the downsample factor).
    """

    def __init__(self, penalty='l1', loss_mult=None, vox_weight=None):
        self.penalty = penalty
        self.loss_mult = loss_mult
        self.vox_weight = vox_weight

    def _diffs(self, y):
        vol_shape = y.get_shape().as_list()[1:-1]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 1
            # permute dimensions to put the ith dimension first
            r = [d, *range(d), *range(d + 1, ndims + 2)]
            yp = K.permute_dimensions(y, r)
            dfi = yp[1:, ...] - yp[:-1, ...]

            if self.vox_weight is not None:
                w = K.permute_dimensions(self.vox_weight, r)
                # TODO: Need to add square root, since for non-0/1 weights this is bad.
                dfi = w[1:, ...] * dfi

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
            df[i] = K.permute_dimensions(dfi, r)

        return df

    def loss(self, _, y_pred):
        """
        returns Tensor of size [bs]
        """

        if self.penalty == 'l1':
            dif = [tf.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]

        df = [tf.reduce_mean(K.batch_flatten(f), axis=-1) for f in dif]
        grad = tf.add_n(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad


# class MutualInformation(ne.metrics.MutualInformation):
#     """
#     Soft Mutual Information approximation for intensity volumes
#
#     More information/citation:
#     - Courtney K Guo.
#       Multi-modal image registration with unsupervised deep learning.
#       PhD thesis, Massachusetts Institute of Technology, 2019.
#     - M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
#       SynthMorph: learning contrast-invariant registration without acquired images
#       IEEE Transactions on Medical Imaging (TMI), in press, 2021
#       https://doi.org/10.1109/TMI.2021.3116879
#     """
#
#     def loss(self, y_true, y_pred):
#         return -self.volumes(y_true, y_pred)


def split_tensor(inp, x_split_times, y_split_times):
    if x_split_times > 1:
        x_splitted = tf.split(inp, x_split_times, axis=1)
        x_splitted_concat = tf.concat(x_splitted, axis=0)
    else:
        x_splitted_concat = inp
    if y_split_times > 1:
        y_splitted = tf.split(x_splitted_concat, y_split_times, axis=2)
        y_splitted_concat = tf.concat(y_splitted, axis=0)
    else:
        y_splitted_concat = x_splitted_concat
    return y_splitted_concat


def loss_G(D_fake_output, G_output, target, train_config, cur_epoch=None):
    if 'loss_mask' in train_config and train_config.loss_mask:
        if hasattr(train_config, 'case_filtering_x_subdivision'):
            assert train_config.case_filtering_x_subdivision == 1
        if hasattr(train_config, 'case_filtering_y_subdivision'):
            assert train_config.case_filtering_y_subdivision == 1
        target = target[:, :, :, :-1]
        loss_mask_from_R = target[:, :, :, -1:]

    # If specified, perform patch filtering
    if train_config.is_training and train_config.case_filtering \
            and cur_epoch >= train_config.case_filtering_starting_epoch:
        assert cur_epoch is not None
        # calculate metric
        if train_config.case_filtering_metric == 'ncc':
            min_ncc_threshold = train_config.case_filtering_cur_mean - \
                                train_config.case_filtering_nsigma * train_config.case_filtering_cur_stdev
            target_clipped = tf.clip_by_value(target, 0, 1)
            G_output_clipped = tf.clip_by_value(G_output, 0, 1)
            if train_config.case_filtering_x_subdivision == 1 and train_config.case_filtering_y_subdivision == 1:
                cur_ncc = tf.stop_gradient(NCC(win=20, eps=1e-3).ncc(target_clipped, G_output_clipped))
                cur_mask = tf.cast(tf.math.greater(cur_ncc, min_ncc_threshold), tf.int32)
                train_config.epoch_filtering_ratio.append(1 - tf.reduce_sum(cur_mask) / cur_mask.get_shape().as_list()[0])

                cur_index = tf.squeeze(tf.where(cur_mask))
                G_output = tf.gather(G_output, cur_index, axis=0)
                target = tf.gather(target, cur_index, axis=0)
                D_fake_output = tf.gather(D_fake_output, cur_index, axis=0)
                if 'loss_mask' in train_config and train_config.loss_mask:
                    loss_mask_from_R = tf.gather(loss_mask_from_R, cur_index, axis=0)
            else:
                G_output_clipped_splitted = split_tensor(G_output_clipped, train_config.case_filtering_x_subdivision,
                                                         train_config.case_filtering_y_subdivision)
                target_clipped_splitted = split_tensor(target_clipped, train_config.case_filtering_x_subdivision,
                                                       train_config.case_filtering_y_subdivision)
                cur_ncc = tf.stop_gradient(NCC(win=20, eps=1e-3).ncc(target_clipped_splitted, G_output_clipped_splitted))
                cur_mask = tf.cast(tf.math.greater(cur_ncc, min_ncc_threshold), tf.int32)
                train_config.epoch_filtering_ratio.append(1 - tf.reduce_sum(cur_mask) / cur_mask.get_shape().as_list()[0])

                cur_index = tf.squeeze(tf.where(cur_mask))
                G_output = tf.gather(G_output, cur_index, axis=0)
                target = tf.gather(target, cur_index, axis=0)

                # for D_fake_output, remove a similar ratio of images (to keep the G-D step ratio consistent)
                n_patches = train_config.case_filtering_x_subdivision*train_config.case_filtering_y_subdivision
                bsz = G_output_clipped.get_shape().as_list()[0]
                case_patch_indices = [[n*bsz+i for n in range(n_patches)] for i in range(bsz)]
                case_patch_indices = [x for y in case_patch_indices for x in y]
                cur_mask_permuted = tf.gather(cur_mask, case_patch_indices, axis=0)
                assert cur_mask_permuted.get_shape().as_list() == cur_mask.get_shape().as_list()
                cur_mask_by_case = tf.reduce_sum(tf.reshape(cur_mask, [bsz, n_patches]), axis=-1)
                cur_D_index = tf.math.top_k(cur_mask_by_case,
                                            k=int(bsz*(1-train_config.epoch_filtering_ratio[-1]) + 0.5)).indices
                D_fake_output = tf.gather(D_fake_output, cur_D_index, axis=0)

                # print("Reduced mask", cur_mask_by_case)
                # print("D Indices", cur_D_index, cur_D_index.get_shape().as_list())
                # print(cur_mask, cur_index, target.get_shape().as_list())

        else:
            print("Unsupported case filtering metric")
            exit(1)

    # apply loss mask from R
    if 'loss_mask' in train_config and train_config.loss_mask:
        G_output = G_output * loss_mask_from_R + tf.stop_gradient(G_output * (1-loss_mask_from_R))
        target = target * loss_mask_from_R + tf.stop_gradient(target * (1-loss_mask_from_R))

    # Then calculate losses
    G_berhu_loss = huber_reverse_loss(pred=G_output, label=target)
    G_tv_loss = tf.reduce_mean(tf.image.total_variation(G_output)) / (train_config.image_size ** 2)
    G_dis_loss = tf.reduce_mean(tf.square(1 - D_fake_output))
    G_total_loss = G_berhu_loss + 0.02 * G_tv_loss + train_config.lamda * G_dis_loss

    return G_total_loss, G_dis_loss, G_berhu_loss


def loss_G_with_R_progressive(D_fake_output, G_output, target, target_transformed, alpha, train_config, cur_epoch=None):
    # (1-alpha) * loss_with_target + alpha * loss_with_target_transformed
    if alpha == 0:
        return loss_G(D_fake_output, G_output, target, train_config, cur_epoch)
    else:
        G_total_loss_with_target, G_dis_loss_with_target, G_berhu_loss_with_target \
            = loss_G(D_fake_output, G_output, target, train_config, cur_epoch)
        G_total_loss_with_target_transformed, G_dis_loss_with_target_transformed, G_berhu_loss_with_target_transformed \
            = loss_G(D_fake_output, G_output, target_transformed, train_config, cur_epoch)
        G_total_loss = (1-alpha) * G_total_loss_with_target + alpha * G_total_loss_with_target_transformed
        G_dis_loss = (1-alpha) * G_dis_loss_with_target + alpha * G_dis_loss_with_target_transformed
        G_berhu_loss = (1-alpha) * G_berhu_loss_with_target + alpha * G_berhu_loss_with_target_transformed
        return G_total_loss, G_dis_loss, G_berhu_loss


def loss_cascaded_R1(R_outputs, fixed, training_config):
    training_config.R_loss_type = training_config.R1_params.R_loss_type
    training_config.lambda_r_tv = training_config.R1_params.lambda_r_tv

    return loss_R_no_gt(R_outputs, fixed, training_config)


def loss_cascaded_R2(R_outputs, fixed, training_config):
    training_config.R_loss_type = training_config.R2_params.R_loss_type
    training_config.lambda_r_tv = training_config.R2_params.lambda_r_tv

    return loss_R_no_gt(R_outputs, fixed, training_config)


def loss_R_flow_only(flow_pred, flow_gt, training_config):
    R_flow_mae_loss = tf.reduce_mean(tf.abs(flow_gt - flow_pred))
    if training_config.lambda_r_tv > 0:
        R_flow_tv_loss = tf.reduce_mean(tf.image.total_variation(flow_pred) / (training_config.image_size ** 2))
    else:
        R_flow_tv_loss = 0
    R_total_loss = R_flow_mae_loss + training_config.lambda_r_tv * R_flow_tv_loss
    return R_total_loss, R_flow_mae_loss


def loss_R_with_gt(R_outputs, fixed, flow_gt, loss_mask, training_config):
    moving_transformed, flow_pred = R_outputs

    if 'loss_mask' in training_config and training_config.loss_mask:
        moving_transformed = moving_transformed[:, :, :, :-1]

    if training_config.boundary_clipping:
        # mask out boundaries in fixed and moving_transformed
        moving_transformed = moving_transformed * loss_mask
        fixed = fixed * loss_mask
        flow_gt = flow_gt * loss_mask

    if training_config.R_loss_type == 'berhu':
        R_structure_loss = huber_reverse_loss(pred=moving_transformed, label=fixed)
    else:
        assert training_config.R_loss_type == 'ncc'
        ncc = NCC(win=20, eps=1e-3)
        R_structure_loss = tf.reduce_mean(ncc.loss(y_true=fixed, y_pred=moving_transformed))
        # ncc1, ncc2 = NCC(win=20), NCC(win=40)
        # R_structure_loss = (tf.reduce_mean(ncc1.loss(y_true=fixed, y_pred=moving_transformed))
        #                     + tf.reduce_mean(ncc2.loss(y_true=fixed, y_pred=moving_transformed))) / 2

    if training_config.lambda_r_mae > 0:
        R_flow_mae_loss = tf.reduce_mean(tf.abs(flow_gt - flow_pred))
        # R_flow_mse_loss = tf.reduce_mean(tf.square(flow_gt - flow_pred))
    else:
        R_flow_mae_loss = 0

    if training_config.lambda_r_tv > 0:
        # R_flow_tv_loss = tf.reduce_mean(tf.image.total_variation(flow_pred) / (training_config.image_size ** 2))
        grad = Grad('l2')
        R_flow_tv_loss = tf.reduce_mean(grad.loss(None, flow_pred))
    else:
        R_flow_tv_loss = 0

    R_total_loss = R_structure_loss + training_config.lambda_r_mae * R_flow_mae_loss \
                   + training_config.lambda_r_tv * R_flow_tv_loss

    return R_total_loss, R_structure_loss, R_flow_mae_loss


def loss_R_no_gt(R_outputs, fixed, training_config):
    moving_transformed, flow_pred = R_outputs
    # print(tf.reduce_max(moving_transformed), tf.reduce_min(moving_transformed), tf.reduce_max(fixed), tf.reduce_min(fixed))

    if training_config.R_loss_type == 'berhu':
        R_structure_loss = huber_reverse_loss(pred=moving_transformed, label=fixed)
    else:
        assert training_config.R_loss_type == 'ncc'
        moving_transformed_clipped = tf.clip_by_value(moving_transformed, 0, 1)
        fixed_clipped = tf.clip_by_value(fixed, 0, 1)
        ncc = NCC(win=20, eps=1e-3)
        R_structure_loss = tf.reduce_mean(ncc.loss(y_true=fixed_clipped, y_pred=moving_transformed_clipped))

    R_flow_tv_loss = 0
    if training_config.lambda_r_tv > 0:
        # R_flow_tv_loss = tf.reduce_mean(tf.image.total_variation(flow_pred) / (training_config.image_size ** 2))
        grad = Grad('l2')
        R_flow_tv_loss = tf.reduce_mean(grad.loss(None, flow_pred))

    R_total_loss = R_structure_loss + training_config.lambda_r_tv * R_flow_tv_loss

    # penalize abs of DVF elements along the batch dim
    if hasattr(training_config, 'lambda_dvf_batch_decay') and training_config.lambda_dvf_batch_decay is not None:
        R_DVF_batch_decay = tf.reduce_mean(tf.math.abs(tf.reduce_mean(flow_pred, axis=0)))
        R_total_loss += training_config.lambda_dvf_batch_decay * R_DVF_batch_decay

    return R_total_loss, R_structure_loss


def color_l1_in_hsl(moving_rgb, fixed_rgb, training_config):
    # Warning: this loss does not work well
    moving_transformed_hsl = rgb2hsl_tf(moving_rgb)
    moving_transformed_hsl = moving_transformed_hsl * tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.constant([1/359, 1, 1]), 0), 0), 0)
    fixed_hsl = rgb2hsl_tf(fixed_rgb)
    fixed_hsl = fixed_hsl * tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.constant([1/359, 1, 1]), 0), 0), 0)

    # TODO: better apply different masks to fixed and moving
    fixed_hsl_mask = tf.stop_gradient(fixed_hsl[:, :, :, 2] < training_config.L_channel_ignore_threshold)
    moving_transformed_hsl = tf.boolean_mask(moving_transformed_hsl, fixed_hsl_mask)
    fixed_hsl = tf.boolean_mask(fixed_hsl, fixed_hsl_mask)

    # TODO: better scale the loss of each img based on the mask
    C_structure_loss = l1_loss(moving_transformed_hsl, fixed_hsl)
    return C_structure_loss


def color_l1_in_yiq(moving_rgb, fixed_rgb, training_config):
    if hasattr(training_config, "C_params"):
        training_config.L_channel_ignore_lower_th = training_config.C_params.L_channel_ignore_lower_th
        training_config.L_channel_ignore_upper_th = training_config.C_params.L_channel_ignore_upper_th

    moving_transformed_yiq = rgb2yiq_tf(moving_rgb)
    fixed_yiq = rgb2yiq_tf(fixed_rgb)
    if training_config.L_channel_ignore_lower_th or training_config.L_channel_ignore_upper_th is not None:
        fixed_yiq_mask = tf.logical_and(fixed_yiq[:, :, :, 0] > training_config.L_channel_ignore_lower_th,
                                        fixed_yiq[:, :, :, 0] < training_config.L_channel_ignore_upper_th)
        moving_yiq_mask = tf.logical_and(moving_transformed_yiq[:, :, :, 0] > training_config.L_channel_ignore_lower_th,
                                         moving_transformed_yiq[:, :, :, 0] < training_config.L_channel_ignore_upper_th)
        yiq_mask = tf.stop_gradient(tf.cast(tf.logical_or(fixed_yiq_mask, moving_yiq_mask), tf.float32))
        # moving_transformed_yiq = tf.boolean_mask(moving_transformed_yiq, fixed_yiq_mask)
        # fixed_yiq = tf.boolean_mask(fixed_yiq, fixed_yiq_mask)
        C_structure_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(moving_transformed_yiq - fixed_yiq) *
                                                        tf.expand_dims(yiq_mask, -1), [1, 2, 3]) / (tf.reduce_sum(yiq_mask, [1, 2]) + 1e-3))
    else:
        C_structure_loss = tf.reduce_mean(tf.abs(moving_transformed_yiq - fixed_yiq))
    return C_structure_loss


def loss_C_no_gt(C_outputs, fixed, training_config):
    moving_transformed, color_params = C_outputs

    if hasattr(training_config, "C_params"):
        training_config.C_loss_type = training_config.C_params.C_loss_type
        training_config.hsv_h_reg_term = training_config.C_params.hsv_h_reg_term
        training_config.hsv_s_reg_term = training_config.C_params.hsv_s_reg_term
        training_config.hsv_v_reg_term = training_config.C_params.hsv_v_reg_term

    if training_config.C_loss_type == 'berhu':
        C_structure_loss = huber_reverse_loss(pred=moving_transformed, label=fixed)
    elif training_config.C_loss_type == 'mae_yiq':
        C_structure_loss = color_l1_in_yiq(moving_transformed, fixed, training_config)
        # print(C_structure_loss)
    else:
        raise NotImplementedError()

    C_total_loss = C_structure_loss

    # add regularization terms
    # note: h is delta h, but s/v are scaling factors
    if training_config.hsv_h_reg_term is not None:
        C_total_loss += training_config.hsv_h_reg_term * tf.reduce_sum(tf.square(color_params[:, 0]))
        # C_total_loss += training_config.hsv_h_reg_term * tf.reduce_sum(tf.abs(color_params[:, 0]))
    if training_config.hsv_s_reg_term is not None:
        C_total_loss += training_config.hsv_s_reg_term * tf.reduce_sum(tf.square((1-color_params[:, 1])))
        # C_total_loss += training_config.hsv_s_reg_term * tf.reduce_sum(tf.abs((1-color_params[:, 1])))
    if training_config.hsv_v_reg_term is not None:
        C_total_loss += training_config.hsv_v_reg_term * tf.reduce_sum(tf.square((1-color_params[:, 2])))
        # C_total_loss += training_config.hsv_s_reg_term * tf.reduce_sum(tf.abs((1-color_params[:, 2])))

    return C_total_loss, C_structure_loss


def loss_C_no_gt_with_D(D_fake_output, C_outputs, fixed, training_config):
    moving_transformed, color_params = C_outputs
    # moving_transformed = C_outputs

    if hasattr(training_config, "C_params"):
        training_config.C_loss_type = training_config.C_params.C_loss_type
        training_config.lamda_C = training_config.C_params.lamda_C
        training_config.hsv_h_reg_term = training_config.C_params.hsv_h_reg_term
        training_config.hsv_s_reg_term = training_config.C_params.hsv_s_reg_term
        training_config.hsv_v_reg_term = training_config.C_params.hsv_v_reg_term

    if training_config.C_loss_type == 'berhu':
        C_structure_loss = huber_reverse_loss(pred=moving_transformed, label=fixed)
    elif training_config.C_loss_type == 'mae_yiq':
        C_structure_loss = color_l1_in_yiq(moving_transformed, fixed, training_config)
    else:
        raise NotImplementedError()

    C_dis_loss = tf.reduce_mean(tf.square(1 - D_fake_output))
    C_total_loss = C_structure_loss + training_config.lamda_C * C_dis_loss

    # add regularization terms
    # note: h is delta h, but s/v are scaling factors
    if training_config.hsv_h_reg_term is not None:
        C_total_loss += training_config.hsv_h_reg_term * tf.reduce_sum(tf.square(color_params[:, 0]))
        # C_total_loss += training_config.hsv_h_reg_term * tf.reduce_sum(tf.abs(color_params[:, 0]))
    if training_config.hsv_s_reg_term is not None:
        C_total_loss += training_config.hsv_s_reg_term * tf.reduce_sum(tf.square((1 - color_params[:, 1])))
        # C_total_loss += training_config.hsv_s_reg_term * tf.reduce_sum(tf.abs((1-color_params[:, 1])))
    if training_config.hsv_v_reg_term is not None:
        C_total_loss += training_config.hsv_v_reg_term * tf.reduce_sum(tf.square((1 - color_params[:, 2])))
        # C_total_loss += training_config.hsv_s_reg_term * tf.reduce_sum(tf.abs((1-color_params[:, 2])))

    return C_total_loss, C_dis_loss, C_structure_loss


def loss_D(D_real_output, D_fake_output):
    D_fake_loss = tf.reduce_mean(tf.square(D_fake_output))
    D_real_loss = tf.reduce_mean(tf.square(1 - D_real_output))
    D_total_loss = D_fake_loss + D_real_loss
    return D_total_loss, D_real_loss, D_fake_loss


def l1_loss(output, target):
    loss = tf.reduce_mean(tf.abs(output - target))
    return loss


def huber_reverse_loss(pred, label, delta=0.2, adaptive=True):
    diff = tf.abs(pred - label)
    if adaptive:
        delta = delta * tf.math.reduce_std(label)  # batch-adaptive
    loss = tf.reduce_mean(tf.cast(diff <= delta, tf.float32) * diff + tf.cast(diff > delta, tf.float32) * (diff**2/2 + delta**2/2) / delta)
    return loss


def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.5):
    pt_1 = tf.where(tf.equal(y_true, 1.0), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0.0), y_pred, tf.zeros_like(y_pred))
    return - tf.reduce_mean(alpha * tf.pow(1 - pt_1, gamma) * tf.math.log(tf.clip_by_value(pt_1, 1e-8, 1)) +
                            (1-alpha) * tf.pow(pt_0, gamma) * tf.math.log(tf.clip_by_value(1 - pt_0, 1e-8, 1)))
