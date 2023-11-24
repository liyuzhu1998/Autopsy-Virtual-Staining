import os

import numpy as np

import batch_utils

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import glob, random, logging

from configobj import ConfigObj
from tensorflow import keras

from batch_utils import ImageTransformationBatchLoader, Her2data_splitter
from models import att_unet_2d, discriminator_2d
from models.aligners import aligner_unet_cvpr2018, aligner_unet_cvpr2018_vJX
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from losses import *
from watcher import Watcher
import ops
import time

def init_parameters():
    tc, vc = ConfigObj(), ConfigObj()

    tc.model_path = 'L:/Regstain_Code/code/stage2_20220727_G&RSeperateTrain_initIter=0/' # set the path to save model
    tc.prev_checkpoint_path = None
    tc.save_every_epoch = True

    # pretrained checkpoints to start from
    tc.G_warmstart_checkpoint = None
    tc.D_warmstart_checkpoint = None
    tc.R_warmstart_checkpoint = None
    assert not (tc.prev_checkpoint_path
                and (tc.G_warmstart_checkpoint or tc.D_warmstart_checkpoint or tc.R_warmstart_checkpoint))

    tc.image_path = 'L:/Pneumonia_Dataset/Second_reg/Training/target/*.mat' # path for training data 
    vc.image_path = 'J:/Pneumonia_Dataset/Second_reg/Validation/target/*.mat' # path for validation data

    def convert_inp_path_from_target(inp_path: str):
        return inp_path.replace('target', 'input')

    tc.convert_inp_path_from_target = convert_inp_path_from_target
    vc.convert_inp_path_from_target = convert_inp_path_from_target

    tc.is_mat, vc.is_mat = True, True  # True for .mat, False for .npy
    tc.data_inpnorm, vc.data_inpnorm = 'norm_by_mean_std', 'norm_by_mean_std'
    tc.channel_start_index, vc.channel_start_index = 0, 0
    tc.channel_end_index, vc.channel_end_index = 2, 2  # exclusive

    # network and loss params
    tc.is_training, vc.is_training = True, False
    tc.image_size, vc.image_size = 256, 256
    tc.num_slices, vc.num_slices = 2, 2
    tc.label_channels, vc.label_channels = 3, 3
    assert tc.channel_end_index - tc.channel_start_index == tc.num_slices
    assert vc.channel_end_index - vc.channel_start_index == vc.num_slices
    tc.n_channels, vc.n_channels = 32, 32
    tc.lamda = 50.0  # adv loss

    tc.nf_enc, vc.nf_enc = [8, 16, 16, 32, 32], [8, 16, 16, 32, 32]  # for aligner
    tc.nf_dec, vc.nf_dec = [32, 32, 32, 32, 32, 16, 16], [32, 32, 32, 32, 32, 16, 16]  # for aligner
    tc.R_loss_type = 'ncc'
    tc.lambda_r_tv = 1.0  # .1    # tv of predicted flow
    tc.gauss_kernal_size = 80
    tc.dvf_clipping = True  # clip DVF to [mu-sigma*dvf_clipping_nsigma, mu+sigma*dvf_clipping_nsigma]
    tc.dvf_clipping_nsigma = 3
    tc.dvf_thresholding = True  # clip DVF to [-dvf_thresholding_distance, dvf_thresholding_distance]
    tc.dvf_thresholding_distance = 30

    # training params
    tc.batch_size, vc.batch_size = 4, 4
    tc.n_shuffle_epoch, vc.n_shuffle_epoch = 500, 5000  # for the batchloader
    tc.initial_alternate_steps = 6000  # train G/D for initial_alternate_steps steps before switching to R for the same # of steps
    tc.valid_steps = 100  # perform validation when D_steps % valid_steps == 0 or at the end of a loop of (train G/D, train R)
    tc.n_threads, vc.n_threads = 2, 2
    tc.q_limit, vc.q_limit = 100, 300
    tc.N_epoch = 150  # number of loops

    tc.tol = 0  # current early stopping patience
    tc.max_tol = 2  # the max-allowed early stopping patience
    tc.min_del = 0  # the lowest acceptable loss value reduction

    # case filtering
    tc.case_filtering = False
    tc.case_filtering_metric = 'ncc'  # 'ncc'
    # divide each patch into case_filtering_x_subdivision patches alone the x dimension for filtering (1 = no division)
    tc.case_filtering_x_subdivision = 2
    tc.case_filtering_y_subdivision = 2
    assert tc.case_filtering_x_subdivision >= 1 and tc.case_filtering_y_subdivision >= 1
    tc.case_filtering_starting_epoch = 2  # case filtering only when epoch >= case_filtering_starting_epoch
    tc.case_filtering_cur_mean, tc.case_filtering_cur_stdev = 0.3757, 0.0654  # for lung elastic (256x256 patch)
    tc.case_filtering_nsigma = 2
    tc.case_filtering_recalc_every_eval = True

    # case filtering for dataloader
    tc.filter_blank, vc.filter_blank = True, True
    tc.filter_threshold, vc.filter_threshold = 0.9515, 0.9515  # 0.9515 for elastic/MT

    # per-pixel loss mask to account for out of the field information brought in by R
    tc.loss_mask, vc.loss_mask = False, False  # True, False

    # training resume parameters
    tc.epoch_begin = 79
    # this overrides tc.epoch_  begin the training schedule; tc.epoch_begin is required for logging
    # set it to None when not used
    tc.iter_begin =  None

    return tc, vc


def run_validation(vc, cur_iter, iterator_valid_bl, G_test_step, D_test_step, R_test_step, wtr, use_tqdm=True):
    valid_G_total_loss_list, valid_G_l1_loss_list, valid_G_ssim_list, valid_G_psnr_list, valid_G_ncc_list, \
    valid_D_real_loss_list, valid_D_fake_loss_list, valid_R_total_loss_list = [], [], [], [], [], [], [], []

    valid_idx_iterator = tqdm(range(vc.q_limit)) if use_tqdm else range(vc.q_limit)
    for i in valid_idx_iterator:
        if not use_tqdm:
            print("Running Validation: {}/{}".format(i+1, vc.q_limit), end='\r')
            if i == vc.q_limit - 1:
                print()
        valid_x, valid_y = next(iterator_valid_bl)
        valid_G_total_loss, valid_G_dis_loss, valid_G_l1_loss, valid_G_ssim, valid_G_psnr, valid_G_ncc = G_test_step(
            valid_x, valid_y, 3)
        valid_D_total_loss, valid_D_real_loss, valid_D_fake_loss, valid_G_output = D_test_step(valid_x, valid_y)
        valid_R_total_loss, _, valid_R_output = R_test_step(valid_x, valid_y)

        if i < 5:
            for j in range(vc.batch_size):
                if tc.label_channels == 1:
                    plt.imsave(tc.model_path + f'/output/iter={cur_iter}_sample={i * vc.batch_size + j}_outputG.jpg',
                               valid_G_output[j, :, :, 0], cmap='gray')
                    plt.imsave(tc.model_path + f'/output/iter={cur_iter}_sample={i * vc.batch_size + j}_outputR.jpg',
                               valid_R_output[0][j, :, :, 0], cmap='gray')
                    plt.imsave(tc.model_path + f'/output/iter={cur_iter}_sample={i * vc.batch_size + j}_target.jpg',
                               valid_y[j, :, :, 0], cmap='gray')
                else:
                    plt.imsave(tc.model_path + f'/output/iter={cur_iter}_sample={i * vc.batch_size + j}_outputG.jpg',
                               np.clip(valid_G_output[j, :, :, :].numpy(), 0, 1))
                    plt.imsave(tc.model_path + f'/output/iter={cur_iter}_sample={i * vc.batch_size + j}_outputR.jpg',
                               np.clip(valid_R_output[0][j, :, :, :].numpy(), 0, 1))
                    plt.imsave(tc.model_path + f'/output/iter={cur_iter}_sample={i * vc.batch_size + j}_target.jpg',
                               valid_y[j, :, :, :].numpy())
                    plt.imsave(tc.model_path + f'/output/iter={cur_iter}_sample={i * vc.batch_size + j}_input0.jpg',
                               -valid_x[j, :, :, 0].numpy(), cmap='gray')

        wtr.check_stop()

        valid_G_total_loss_list.append(valid_G_total_loss)
        valid_G_l1_loss_list.append(valid_G_l1_loss)
        valid_G_ssim_list.append(valid_G_ssim)
        valid_G_psnr_list.append(valid_G_psnr)
        valid_G_ncc_list.append(valid_G_ncc.numpy().item())
        if valid_G_ncc.numpy().item() > 1:
            plt.imsave(
                tc.model_path + f'/output/wrongncc{valid_G_ncc.numpy().item()}_iter={cur_iter}_sample={i * vc.batch_size + j}_output.jpg',
                np.clip(valid_G_output[j, :, :, :].numpy(), 0, 1))
            plt.imsave(
                tc.model_path + f'/output/wrongncc{valid_G_ncc.numpy().item()}_iter={cur_iter}_sample={i * vc.batch_size + j}_label.jpg',
                valid_y[j, :, :, :].numpy())
        valid_D_real_loss_list.append(valid_D_real_loss)
        valid_D_fake_loss_list.append(valid_D_fake_loss)
        valid_R_total_loss_list.append(valid_R_total_loss)

    valid_G_total_loss_mean = np.mean(np.array(valid_G_total_loss_list))
    valid_G_l1_loss_mean = np.mean(np.array(valid_G_l1_loss_list))
    valid_G_ssim_mean = np.mean(np.array(valid_G_ssim_list))
    valid_G_psnr_mean = np.mean(np.array(valid_G_psnr_list))
    vaild_G_ncc_mean = np.mean(np.array(valid_G_ncc_list))
    valid_G_ncc_std = np.std(np.array(valid_G_ncc_list))
    valid_D_real_loss_mean = np.mean(np.array(valid_D_real_loss_list))
    valid_D_fake_loss_mean = np.mean(np.array(valid_D_fake_loss_list))
    valid_R_total_loss_mean = np.mean(np.array(valid_R_total_loss_list))

    return valid_G_total_loss_mean, valid_G_l1_loss_mean, valid_G_ssim_mean, valid_G_psnr_mean, \
           vaild_G_ncc_mean, valid_G_ncc_std, valid_D_real_loss_mean, valid_D_fake_loss_mean, valid_R_total_loss_mean


if __name__ == '__main__':
    wtr = Watcher()
    tc, vc = init_parameters()

    tf.io.gfile.mkdir(tc.model_path)
    tf.io.gfile.mkdir(tc.model_path + '/output')

    # ======================= input pipeline =========================

    train_images = glob.glob(tc.image_path)
    valid_images = glob.glob(vc.image_path)
    random.shuffle(train_images)
    random.shuffle(valid_images)

    ops.copy_code(model_path=tc.model_path)

    print('Number of train images: ' + str(len(train_images)))
    print('Train images examples: ' + str(train_images[:5]))
    print('Length of valid images: ' + str(len(valid_images)))
    print('Valid images examples: ' + str(valid_images[:5]))

    train_bl = ImageTransformationBatchLoader(train_images, tc, tc.num_slices, is_testing=False,
                                              n_parallel_calls=tc.n_threads, q_limit=tc.q_limit,
                                              n_epoch=tc.n_shuffle_epoch)
    valid_bl = ImageTransformationBatchLoader(valid_images, vc, vc.num_slices, is_testing=True,
                                              n_parallel_calls=vc.n_threads, q_limit=vc.q_limit,
                                              n_epoch=vc.n_shuffle_epoch)

    iterator_train_bl = iter(train_bl.dataset)
    iterator_valid_bl = iter(valid_bl.dataset)

    # ======================= model =========================
    model_G = att_unet_2d((tc.image_size, tc.image_size, tc.num_slices), n_labels=tc.label_channels,
                          filter_num=[tc.n_channels, tc.n_channels * 2, tc.n_channels * 4, tc.n_channels * 8,
                                      tc.n_channels * 16],
                          stack_num_down=3, stack_num_up=3, activation='LeakyReLU',
                          atten_activation='ReLU', attention='add',
                          output_activation=None, batch_norm=True, pool='ave', unpool='bilinear', name='model_G')

    model_D = discriminator_2d((tc.image_size, tc.image_size, tc.label_channels),
                               filter_num=[tc.n_channels, tc.n_channels * 2, tc.n_channels * 4, tc.n_channels * 8,
                                           tc.n_channels * 16],
                               stack_num_down=1, activation='LeakyReLU',
                               batch_norm=False, pool=False, name='model_D')

    model_R = aligner_unet_cvpr2018_vJX([tc.image_size, tc.image_size], tc.nf_enc, tc.nf_dec,
                                        gauss_kernal_size=tc.gauss_kernal_size,
                                        flow_clipping=tc.dvf_clipping,
                                        flow_clipping_nsigma=tc.dvf_clipping_nsigma,
                                        flow_thresholding=tc.dvf_thresholding,
                                        flow_thresh_dis=tc.dvf_thresholding_distance,
                                        loss_mask=tc.loss_mask, loss_mask_from_prev_cascade=False)

    if tc.G_warmstart_checkpoint:
        assert tc.D_warmstart_checkpoint is not None
        model_G.load_weights(tc.G_warmstart_checkpoint)
        model_D.load_weights(tc.D_warmstart_checkpoint)
        print("Loaded G checkpoints from", tc.G_warmstart_checkpoint)
        print("Loaded D checkpoints from", tc.D_warmstart_checkpoint)
    if tc.R_warmstart_checkpoint:
        model_R.load_weights(tc.R_warmstart_checkpoint)
        print("Loaded R checkpoints from", tc.R_warmstart_checkpoint)

    G_optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    D_optimizer = keras.optimizers.Adam(learning_rate=1e-5)
    R_optimizer = keras.optimizers.Adam(learning_rate=1e-4)

    if tc.prev_checkpoint_path:
        model_G.load_weights(tc.prev_checkpoint_path + '/model_G_latest.h5')
        model_D.load_weights(tc.prev_checkpoint_path + '/model_D_latest.h5')
        model_R.load_weights(tc.prev_checkpoint_path + '/model_R_latest.h5')
        G_optimizer.set_weights(tc.prev_checkpoint_path + '/optimizer_G_latest.npy')
        D_optimizer.set_weights(tc.prev_checkpoint_path + '/optimizer_D_latest.npy')
        R_optimizer.set_weights(tc.prev_checkpoint_path + '/optimizer_R_latest.npy')
        print("Loaded latest weight and optimizer checkpoints from", tc.prev_checkpoint_path)


    def G_train_step(input_image, target, epoch):
        with tf.GradientTape() as gen_tape:
            G_outputs = model_G(input_image, training=True)
            D_fake_output = model_D(G_outputs, training=True)

            if epoch > 0:
                if tc.dvf_clipping or tc.dvf_thresholding:
                    target_transformed, _, _ = model_R([target, G_outputs])
                else:
                    start = time.perf_counter()
                    time.sleep(1)
                    target_transformed, _ = model_R([target, G_outputs])
                    end = time.perf_counter()
                    elapsed = end-start
            else:
                target_transformed = target

            G_total_loss, G_dis_loss, G_l1_loss = loss_G(D_fake_output, G_outputs, target_transformed, train_config=tc, cur_epoch=epoch)

        G_gradients = gen_tape.gradient(G_total_loss, model_G.trainable_variables,
                                        unconnected_gradients=tf.UnconnectedGradients.ZERO)
        G_optimizer.apply_gradients(zip(G_gradients, model_G.trainable_variables))
        return G_total_loss, G_dis_loss, G_l1_loss, G_outputs, target_transformed


    def G_test_step(input_image, target, epoch):
        G_outputs = model_G(input_image, training=False)
        G_outputs_clipped_splitted = split_tensor(tf.clip_by_value(G_outputs, 0, 1), tc.case_filtering_x_subdivision,
                                                  tc.case_filtering_y_subdivision)
        target_clipped_splitted = split_tensor(tf.clip_by_value(target, 0, 1), tc.case_filtering_x_subdivision,
                                               tc.case_filtering_y_subdivision)
        D_fake_output = model_D(G_outputs, training=False)

        G_total_loss, G_dis_loss, G_l1_loss = loss_G(D_fake_output, G_outputs, target, train_config=tc, cur_epoch=epoch)
        G_ssim = tf.image.ssim(G_outputs, target, max_val=1)
        G_psnr = tf.image.psnr(G_outputs, target, max_val=1)
        G_ncc = tf.reduce_mean(NCC(win=20, eps=1e-3).ncc(target_clipped_splitted, G_outputs_clipped_splitted))

        return G_total_loss, G_dis_loss, G_l1_loss, G_ssim, G_psnr, G_ncc


    def D_train_step(input_image, target):
        with tf.GradientTape() as disc_tape:
            G_outputs = tf.stop_gradient(model_G(input_image, training=True))
            D_real_output = model_D(target, training=True)
            D_fake_output = model_D(G_outputs, training=True)
            D_total_loss, D_real_loss, D_fake_loss = loss_D(D_real_output, D_fake_output)

        D_gradients = disc_tape.gradient(D_total_loss, model_D.trainable_variables)
        D_optimizer.apply_gradients(zip(D_gradients, model_D.trainable_variables))
        return D_total_loss, D_real_loss, D_fake_loss, G_outputs


    def D_test_step(input_image, target):
        G_outputs = model_G(input_image, training=False)
        D_real_output = model_D(target, training=False)
        D_fake_output = model_D(G_outputs, training=False)
        D_total_loss, D_real_loss, D_fake_loss = loss_D(D_real_output, D_fake_output)
        return D_total_loss, D_real_loss, D_fake_loss, G_outputs


    def R_train_step(input_image, target):
        with tf.GradientTape() as reg_tape:
            G_outputs = tf.stop_gradient(model_G(input_image, training=True))
            R_outputs = model_R([target, G_outputs], training=True)
            if tc.dvf_clipping or tc.dvf_thresholding:
                R_outputs = R_outputs[:-1]
            R_total_loss, R_berhu_loss = loss_R_no_gt(R_outputs, G_outputs, tc)

        R_gradients = reg_tape.gradient(R_total_loss, model_R.trainable_variables)
        R_optimizer.apply_gradients(zip(R_gradients, model_R.trainable_variables))
        return R_total_loss, R_berhu_loss, R_outputs


    def R_test_step(input_image, target):
        G_outputs = model_G(input_image, training=False)
        R_outputs = model_R([target, G_outputs])
        if tc.dvf_clipping or tc.dvf_thresholding:
            R_outputs = R_outputs[:-1]
        R_total_loss, R_berhu_loss = loss_R_no_gt(R_outputs, G_outputs, tc)
        return R_total_loss, R_berhu_loss, R_outputs


    min_loss = 1e8
    max_psnr = 0
    epoch_begin = tc.epoch_begin
    iter_D_count = 0
    warmstart_first_epoch_elapsed_iters = None
    if tc.iter_begin is not None:
        iter_D_count += tc.iter_begin
        assert epoch_begin is not None
        warmstart_first_epoch_elapsed_iters = tc.iter_begin
        for i in range(epoch_begin):
            warmstart_first_epoch_elapsed_iters -= max(int(tc.initial_alternate_steps * 0.9 ** i), 500)
    elif epoch_begin is not None:
        for i in range(epoch_begin):
            iter_D_count += max(int(tc.initial_alternate_steps * 0.9 ** i), 500)
    print("Training from iteration", iter_D_count)

    # loop over epoches
    for epoch in range(epoch_begin, tc.N_epoch):

        print(f'Current iter_D_count: {iter_D_count}')

        train_G_total_loss_list, train_G_l1_loss_list, train_R_total_loss_list = [], [], []

        # loop over batches
        if warmstart_first_epoch_elapsed_iters is None or warmstart_first_epoch_elapsed_iters == 0:
            num_checkpoint_D = max(int(tc.initial_alternate_steps * 0.9 ** epoch), 500)
            num_checkpoint_R = max(int(tc.initial_alternate_steps * 0.9 ** epoch), 500)
        else:
            num_checkpoint_D = warmstart_first_epoch_elapsed_iters
            num_checkpoint_R = warmstart_first_epoch_elapsed_iters
            warmstart_first_epoch_elapsed_iters = None

        if epoch > 0 or (tc.G_warmstart_checkpoint is None and tc.prev_checkpoint_path is None):
            print('training G & D ...')
            tc.epoch_filtering_ratio = []
            if tc.case_filtering and epoch >= tc.case_filtering_starting_epoch:
                format_str = "Train G Case Filtering {} for epoch {} with threshold {} and {}x{} subdivision"
                print(format_str.format(tc.case_filtering, epoch,
                                        tc.case_filtering_cur_mean - tc.case_filtering_nsigma * tc.case_filtering_cur_stdev,
                                        tc.case_filtering_x_subdivision, tc.case_filtering_y_subdivision))
            else:
                print("Train G Case Filtering False for epoch {}".format(epoch))

            for i in tqdm(range(num_checkpoint_D)):

                start = time.perf_counter()
                time.sleep(1)

                NumGen = max(3, int(12 - iter_D_count // 4000))

                for j in range(NumGen):
                    # selecting samples for the current batch
                    train_x, train_y = next(iterator_train_bl)

                    train_G_total_loss, train_G_dis_loss, train_G_l1_loss, train_G_output, train_R_output = G_train_step(
                        train_x, train_y, epoch)
                    train_G_total_loss_list.append(train_G_total_loss)
                    train_G_l1_loss_list.append(train_G_l1_loss)
                    wtr.check_stop()

                if i % 1000 == 0:
                    plt.imsave(tc.model_path + f'/output/train_epoch={epoch}_sample={i}_outputG.jpg',
                               np.clip(train_G_output[0, :, :, :].numpy(), 0, 1))
                    plt.imsave(tc.model_path + f'/output/train_epoch={epoch}_sample={i}_outputR.jpg',
                               np.clip(train_R_output[0, :, :, :].numpy(), 0, 1))
                    plt.imsave(tc.model_path + f'/output/train_epoch={epoch}_sample={i}_target.jpg',
                               train_y[0, :, :, :].numpy())
                    plt.imsave(tc.model_path + f'/output/train_epoch={epoch}_sample={i}_input0.jpg',
                               -train_x[0, :, :, 0].numpy(), cmap='gray')

                train_x, train_y = next(iterator_train_bl)
                train_D_total_loss, train_D_real_loss, train_D_fake_loss, _ = D_train_step(train_x, train_y)
                iter_D_count += 1

                end = time.perf_counter() 
                elapsed = end-start

                #######################################################################################################
                # mid-round validation
                #######################################################################################################
                if iter_D_count % tc.valid_steps == 0 and i != num_checkpoint_D-1:
                    valid_G_total_loss_mean, valid_G_l1_loss_mean, valid_G_ssim_mean, valid_G_psnr_mean, \
                        vaild_G_ncc_mean, valid_G_ncc_std, valid_D_real_loss_mean, valid_D_fake_loss_mean, \
                        valid_R_total_loss_mean = run_validation(vc, iter_D_count, iterator_valid_bl, G_test_step,
                                                                 D_test_step, R_test_step, wtr, use_tqdm=False)

                    train_G_total_loss_mean = np.mean(np.array(train_G_total_loss_list))
                    train_G_l1_loss_mean = np.mean(np.array(train_G_l1_loss_list))
                    train_R_total_loss_mean = np.mean(np.array(train_R_total_loss_list))
                    train_G_total_loss_list, train_G_l1_loss_list, train_R_total_loss_list = [], [], []

                    # update case filtering metrics
                    # if vaild_G_ncc_mean < 0.5:
                    if tc.case_filtering and tc.case_filtering_recalc_every_eval:
                        tc.case_filtering_cur_mean, case_filtering_cur_stdev = vaild_G_ncc_mean, valid_G_ncc_std
                        print("Updated filtering mean to {} and stdev to {}".format(vaild_G_ncc_mean, valid_G_ncc_std))

                    # output log
                    text = ["round ", "iter_D_count ", "train_G_total_loss_mean ", "train_G_l1_loss_mean ",
                            "valid_G_total_loss_mean ", "valid_G_l1_loss_mean ",
                            "valid_G_ssim_mean", "valid_G_psnr_mean", "valid_G_ncc_mean ", "valid_G_ncc_std ",
                            "valid_D_real_loss_mean ", "valid_D_fake_loss_mean ", "valid_R_total_loss_mean "]
                    value = [epoch, iter_D_count, train_G_total_loss_mean, train_G_l1_loss_mean,
                             valid_G_total_loss_mean, valid_G_l1_loss_mean,
                             valid_G_ssim_mean, valid_G_psnr_mean, vaild_G_ncc_mean, valid_G_ncc_std,
                             valid_D_real_loss_mean, valid_D_fake_loss_mean, valid_R_total_loss_mean]

                    msg = ops.verbose_msg(text, value, json_format=True)
                    ops.print_and_save_msg(msg + '\n', tc.model_path + '/train_log.txt')

                    # save_model
                    if min_loss - valid_G_l1_loss_mean > tc.min_del or valid_G_psnr_mean > max_psnr \
                            or tc.save_every_epoch:
                        # tol = 0  # refresh early stopping patience
                        model_G.save_weights(tc.model_path + f'/model_G_iter={iter_D_count}.h5')
                        model_D.save_weights(tc.model_path + f'/model_D_iter={iter_D_count}.h5')
                        model_R.save_weights(tc.model_path + f'/model_R_iter={iter_D_count}.h5')

                        if min_loss - valid_G_l1_loss_mean > tc.min_del:
                            print('Validation loss is improved from {} to {}'.format(min_loss, valid_G_l1_loss_mean))
                            min_loss = valid_G_l1_loss_mean  # update the loss record

                        if valid_G_psnr_mean > max_psnr:
                            print('Validation PSNR is improved from {} to {}'.format(max_psnr, valid_G_psnr_mean))
                            max_psnr = valid_G_psnr_mean

                    model_G.save_weights(tc.model_path + f'/model_G_latest.h5')
                    model_D.save_weights(tc.model_path + f'/model_D_latest.h5')
                    model_R.save_weights(tc.model_path + f'/model_R_latest.h5')
                    np.save(tc.model_path + f'/optimizer_G_latest.npy', G_optimizer.get_weights())
                    np.save(tc.model_path + f'/optimizer_D_latest.npy', D_optimizer.get_weights())
                    np.save(tc.model_path + f'/optimizer_R_latest.npy', R_optimizer.get_weights())

        print('training R ...')
        for i in tqdm(range(num_checkpoint_R)):
            start = time.perf_counter()
            time.sleep(1)
            train_x, train_y = next(iterator_train_bl)
            train_R_total_loss, _, train_R_output = R_train_step(train_x, train_y)
            wtr.check_stop()
            train_R_total_loss_list.append(train_R_total_loss)
            end = time.perf_counter() 
            elapsed = end-start

        ###############################################################################################################
        # round-end validation
        ###############################################################################################################
        valid_G_total_loss_mean, valid_G_l1_loss_mean, valid_G_ssim_mean, valid_G_psnr_mean, \
            vaild_G_ncc_mean, valid_G_ncc_std, valid_D_real_loss_mean, valid_D_fake_loss_mean, \
            valid_R_total_loss_mean = \
            run_validation(vc, iter_D_count, iterator_valid_bl, G_test_step, D_test_step, R_test_step, wtr)

        train_G_total_loss_mean = np.mean(np.array(train_G_total_loss_list))
        train_G_l1_loss_mean = np.mean(np.array(train_G_l1_loss_list))
        train_R_total_loss_mean = np.mean(np.array(train_R_total_loss_list))
        train_G_total_loss_list, train_G_l1_loss_list, train_R_total_loss_list = [], [], []

        # update case filtering metrics
        # if vaild_G_ncc_mean < 0.5:
        if tc.case_filtering and tc.case_filtering_recalc_every_eval:
            tc.case_filtering_cur_mean, case_filtering_cur_stdev = vaild_G_ncc_mean, valid_G_ncc_std
            print("Updated filtering mean to {} and stdev to {}".format(vaild_G_ncc_mean, valid_G_ncc_std))

        # output log
        text = ["round ", "iter_D_count ", "train_G_total_loss_mean ", "train_G_l1_loss_mean ",
                "valid_G_total_loss_mean ", "valid_G_l1_loss_mean ",
                "valid_G_ssim_mean", "valid_G_psnr_mean", "valid_G_ncc_mean ", "valid_G_ncc_std ",
                "valid_D_real_loss_mean ", "valid_D_fake_loss_mean ", "valid_R_total_loss_mean "]
        value = [epoch, iter_D_count, train_G_total_loss_mean, train_G_l1_loss_mean,
                 valid_G_total_loss_mean, valid_G_l1_loss_mean,
                 valid_G_ssim_mean, valid_G_psnr_mean, vaild_G_ncc_mean, valid_G_ncc_std,
                 valid_D_real_loss_mean, valid_D_fake_loss_mean, valid_R_total_loss_mean]

        msg = ops.verbose_msg(text, value, json_format=True)
        ops.print_and_save_msg(msg + '\n', tc.model_path + '/train_log.txt')

        # save_model
        if min_loss - valid_G_l1_loss_mean > tc.min_del or valid_G_psnr_mean > max_psnr \
                or tc.save_every_epoch:
            # tol = 0  # refresh early stopping patience
            model_G.save_weights(tc.model_path + f'/model_G_round={epoch}.h5')
            model_D.save_weights(tc.model_path + f'/model_D_round={epoch}.h5')
            model_R.save_weights(tc.model_path + f'/model_R_round={epoch}.h5')

            if min_loss - valid_G_l1_loss_mean > tc.min_del:
                print('Validation loss is improved from {} to {}'.format(min_loss, valid_G_l1_loss_mean))
                min_loss = valid_G_l1_loss_mean  # update the loss record

            if valid_G_psnr_mean > max_psnr:
                print('Validation PSNR is improved from {} to {}'.format(min_loss, valid_G_psnr_mean))
                max_psnr = valid_G_psnr_mean

        model_G.save_weights(tc.model_path + f'/model_G_latest.h5')
        model_D.save_weights(tc.model_path + f'/model_D_latest.h5')
        model_R.save_weights(tc.model_path + f'/model_R_latest.h5')
        np.save(tc.model_path + f'/optimizer_G_latest.npy', G_optimizer.get_weights())
        np.save(tc.model_path + f'/optimizer_D_latest.npy', D_optimizer.get_weights())
        np.save(tc.model_path + f'/optimizer_R_latest.npy', R_optimizer.get_weights())
