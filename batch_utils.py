import tensorflow as tf
import numpy as np
from scipy import ndimage, misc
from scipy.io import loadmat
from tensorflow_addons import image
from tensorflow_addons.image import translate
import random
import glob

from matplotlib import pyplot as plt


class BatchLoader(object):
    def __init__(self, images, config, input_channels, is_testing, n_parallel_calls, q_limit, n_epoch):
        self.num_epoch = n_epoch
        self.images = images
        self.PATHS = []
        self.config = config
        self.is_testing = is_testing
        self.input_channels = input_channels
        self.label_channels = config.label_channels
        self.raw_size = config.image_size
        self.image_size = config.image_size
        self.num_parallel_calls = n_parallel_calls
        self.q_limit = q_limit

        for i in range(self.num_epoch):
            if self.config.is_training:
                random.shuffle(self.images)
            self.PATHS.extend(self.images)
        print("Length of the image file list: " + str(len(self.PATHS)))

        self.dataset = self.create_dataset_from_generator()

        assert (self.is_testing and not self.config.is_training) or (not self.is_testing and self.config.is_training)
        if self.config.is_training:
            self.dataset = self.dataset.shuffle(buffer_size=self.q_limit, reshuffle_each_iteration=True)
            self.dataset = self.dataset.map(self.augment, num_parallel_calls=config.n_threads)

        # self.dataset = self.dataset.map(lambda x, y: self._preprocessing(x, y), num_parallel_calls=config.n_threads)

        self.dataset = self.dataset.batch(self.config.batch_size).prefetch(100)
        self.iter: tf.compat.v1.data.Iterator = tf.compat.v1.data.Iterator.from_structure(
            tf.compat.v1.data.get_output_types(self.dataset),
            tf.compat.v1.data.get_output_shapes(self.dataset))

    def create_dataset_from_generator(self):
        raise NotImplementedError

    def augment(self, *args):
        raise NotImplementedError

    def init_iter(self):
        return self.iter.make_initializer(self.dataset)


class ImageTransformationBatchLoader(BatchLoader):
    def __init__(self, train_images, tc, num_slices, **kwargs):
        super().__init__(train_images, tc, num_slices, **kwargs)

        if not hasattr(self.config, 'filter_blank'):
            self.config.filter_blank = False
        elif self.config.filter_blank:
            assert self.config.filter_threshold is not None

        self.cur_filter_count = 0
        self.case_trial_limit = 10

    def create_dataset_from_generator(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.PATHS)
        return dataset.interleave(lambda x: tf.data.Dataset.from_generator(self.parse_and_generate, output_types=(tf.float32, tf.float32),
                                                                           args=(x,),
                                                                           output_shapes=(tf.TensorShape([self.image_size, self.image_size, self.input_channels]),
                                                                                          tf.TensorShape([self.image_size, self.image_size, self.label_channels]))),
                                  cycle_length=self.config.n_threads, block_length=1, num_parallel_calls=self.config.n_threads)

    def parse_and_generate(self, path):

        s = self.config.image_size
        stride = s

        start, end = self.config.channel_start_index, self.config.channel_end_index
        path = path.decode('UTF-8')

        if self.config.is_mat:
            # image = loadmat(path.replace('target', 'input'))['input'].astype(np.float32)[:, :, start:end]
            try:
                image = loadmat(self.config.convert_inp_path_from_target(path))['input'].astype(np.float32)[:, :, start:end]
                label = loadmat(path)['target'].astype(np.float32) #/ 255.0
            except:
                print(self.config.convert_inp_path_from_target(path))
        else:
            # image = np.transpose(np.load(path.replace('target', 'input')).astype(np.float32)[start:end, :, :], axes=[1, 2, 0])
            image = np.transpose(
                np.load(self.config.convert_inp_path_from_target(path)).astype(np.float32)[:, :, start:end],
                axes=[1, 2, 0])
            label = np.transpose(np.load(path).astype(np.float32), axes=[1, 2, 0]) #/ 255.0

        if self.config.data_inpnorm == 'norm_by_specified_value':
            normalize_vector = [1500, 1500, 1500, 1000]
            normalize_vector = np.reshape(normalize_vector, [1, 1, 4])
            image = image / normalize_vector
        elif self.config.data_inpnorm == 'norm_by_mean_std':
            image = (image - np.mean(image)) / (np.std(image) + 1e-5)

        crop_edge = 30
        image = image[crop_edge:-crop_edge, crop_edge:-crop_edge, :]
        label = label[crop_edge:-crop_edge, crop_edge:-crop_edge, :]

        size = image.shape[0]

        cur_trial_count = 0
        x = 0
        while True:
            y = 0
            while True:
                rand_choice_stride = random.randint(0, 15)
                xx = min(x + rand_choice_stride * s // 16, size - s)
                yy = min(y + rand_choice_stride * s // 16, size - s)
                if yy != size - s and xx != size - s:
                    img = image[xx:xx + s, yy:yy + s]
                    lab = label[xx:xx + s, yy:yy + s]

                    if self.config.filter_blank and np.mean(lab) >= self.config.filter_threshold \
                            and cur_trial_count < self.case_trial_limit:
                        # print("debug: fitered out patch with mean:", np.mean(lab))
                        # self.cur_filter_count += 1
                        # plt.imsave('filtered_label_patch_{}.jpg'.format(self.cur_filter_count), lab)
                        cur_trial_count += 1
                        # if cur_trial_count == self.case_trial_limit:
                        #     print("Blank filtering max trial reached")
                        continue
                    else:
                        # if 0 < cur_trial_count < self.case_trial_limit:
                        #     print("Blank filtering helps +1")
                        yield (img.astype(np.float32), lab.astype(np.float32))

                if yy == size - s:
                    break
                y += stride
            if xx == size - s:
                break
            x += stride

    def augment(self, img, lab):
        imglab = tf.concat([img, lab], axis=-1)
        imglab = tf.image.random_flip_left_right(imglab)

        img = imglab[:, :, 0:self.config.num_slices]
        lab = imglab[:, :, self.config.num_slices:self.config.num_slices+self.label_channels]

        random_number = tf.random.uniform(shape=(), minval=0, maxval=4, dtype=tf.int32)
        img = tf.image.rot90(img, k=random_number)
        lab = tf.image.rot90(lab, k=random_number)

        # img = img + tf.random.truncated_normal(mean=0, stddev=0.05,
        #                                        shape=[self.image_size, self.image_size, self.config.num_slices])

        return img, lab

'''
class ImageTransformationBatchLoader(BatchLoader):
    def create_dataset_from_generator(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.PATHS)
        return dataset.interleave(lambda x: tf.data.Dataset.from_generator(self.parse_and_generate, output_types=(tf.float32, tf.float32),
                                                                           args=(x,),
                                                                           output_shapes=(tf.TensorShape([self.image_size, self.image_size, self.input_channels]),
                                                                                          tf.TensorShape([self.image_size, self.image_size, self.label_channels]))),
                                  cycle_length=self.config.n_threads, block_length=1, num_parallel_calls=self.config.n_threads)

    def parse_and_generate(self, path):

        s = self.config.image_size
        stride = s

        start, end = self.config.channel_start_index, self.config.channel_end_index
        path = path.decode('UTF-8')

        if self.config.is_mat:
            # image = loadmat(path.replace('target', 'input'))['input'].astype(np.float32)[:, :, start:end]
            image = loadmat(self.config.convert_inp_path_from_target(path))['input'].astype(np.float32)[:, :, start:end]
            label = loadmat(path)['target'].astype(np.float32) / 255.0
        else:
            # image = np.transpose(np.load(path.replace('target', 'input')).astype(np.float32)[start:end, :, :], axes=[1, 2, 0])
            image = np.transpose(np.load(self.config.convert_inp_path_from_target(path)).astype(np.float32)[start:end, :, :], axes=[1, 2, 0])
            label = np.transpose(np.load(path).astype(np.float32), axes=[1, 2, 0]) / 255.0

        if self.config.data_inpnorm:
            normalize_vector = [1500, 1500, 1500, 1000]
            normalize_vector = np.reshape(normalize_vector, [1, 1, 4])
            image = image / normalize_vector

        crop_edge = 30
        image = image[crop_edge:-crop_edge, crop_edge:-crop_edge, :]
        label = label[crop_edge:-crop_edge, crop_edge:-crop_edge, :]

        size = image.shape[0]

        x = 0
        while True:
            y = 0
            while True:
                rand_choice_stride = random.randint(0, 15)
                xx = min(x + rand_choice_stride * s // 16, size - s)
                yy = min(y + rand_choice_stride * s // 16, size - s)
                if yy != size - s and xx != size - s:
                    img = image[xx:xx + s, yy:yy + s]
                    lab = label[xx:xx + s, yy:yy + s]

                    yield (img.astype(np.float32), lab.astype(np.float32))

                if yy == size - s:
                    break
                y += stride
            if xx == size - s:
                break
            x += stride

    def augment(self, img, lab):
        imglab = tf.concat([img, lab], axis=-1)
        imglab = tf.image.random_flip_left_right(imglab)

        img = imglab[:, :, 0:self.config.num_slices]
        lab = imglab[:, :, self.config.num_slices:self.config.num_slices+self.label_channels]

        random_number = tf.random.uniform(shape=(), minval=0, maxval=4, dtype=tf.int32)
        img = tf.image.rot90(img, k=random_number)
        lab = tf.image.rot90(lab, k=random_number)

        # img = img + tf.random.truncated_normal(mean=0, stddev=0.05,
        #                                        shape=[self.image_size, self.image_size, self.config.num_slices])

        return img, lab
'''

class ImageTransformationBatchLoader_Testing(BatchLoader):

    def create_dataset_from_generator(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.PATHS)
        return dataset.interleave(lambda x: tf.data.Dataset.from_generator(self.parse_and_generate, output_types=(tf.float32, tf.float32),
                                                                           args=(x,),
                                                                           output_shapes=(tf.TensorShape([self.image_size, self.image_size, self.input_channels]),
                                                                                          tf.TensorShape([self.image_size, self.image_size, self.label_channels]))),
                                  cycle_length=self.config.n_threads, block_length=1, num_parallel_calls=self.config.n_threads)

    def parse_and_generate(self, path):

        s = self.config.image_size
        stride = s

        start, end = self.config.channel_start_index, self.config.channel_end_index
        path = path.decode('UTF-8')

        if self.config.is_mat:
            image = loadmat(self.config.convert_inp_path_from_target(path))['input'].astype(np.float32)[:, :, start:end]
            label = loadmat(path)['target'].astype(np.float32) / 255
        else:
            image = np.transpose(np.load(self.config.convert_inp_path_from_target(path)).astype(np.float32)[start:end, :, :], axes=[1, 2, 0])
            label = np.transpose(np.load(path).astype(np.float32), axes=[1, 2, 0]) / 255.0

        if self.config.data_inpnorm == 'norm_by_specified_value':
            normalize_vector = [1500, 1500, 1500, 1000]
            normalize_vector = np.reshape(normalize_vector, [1, 1, 4])
            image = image / normalize_vector
        elif self.config.data_inpnorm == 'norm_by_mean_std':
            image = (image - np.mean(image)) / (np.std(image) + 1e-5)

        yield (image.astype(np.float32), label.astype(np.float32))


class ImageRegistrationBatchLoader(BatchLoader):
    # implement `create_dataset_from_generator` (yielding moving, fixed, and flow)
    # and `augment` (taking moving, fixed, and flow)
    def create_dataset_from_generator(self):
        return tf.data.Dataset.from_generator(self.parse_and_generate,
                                              output_types=(tf.float32, tf.float32, tf.float32, tf.float32),
                                              output_shapes=(tf.TensorShape([self.image_size, self.image_size, 3]),
                                                             tf.TensorShape([self.image_size, self.image_size, 3]),
                                                             tf.TensorShape([self.image_size, self.image_size, 2]),
                                                             tf.TensorShape([self.image_size, self.image_size, 1])))

    def parse_and_generate(self):
        s = self.config.image_size
        stride = s
        for path in self.PATHS:
            # print(path)
            if self.config.is_mat:
                moving = loadmat(path)['target'].astype(np.float32)
            else:
                moving = np.load(path).astype(np.float32)

            # generate a random flow
            maxshift = 40
            x_shift = int(random.uniform(-maxshift, maxshift))
            y_shift = int(random.uniform(-maxshift, maxshift))
            flow_gt = np.stack([np.ones([s, s]) * x_shift, np.ones([s, s]) * y_shift], axis=-1)
            loss_mask = np.ones((s, s, 1))
            if x_shift >= 0:
                loss_mask[:x_shift, :, :] = 0
            else:
                loss_mask[x_shift:, :, :] = 0
            if y_shift >= 0:
                loss_mask[:, :y_shift, :] = 0
            else:
                loss_mask[:, y_shift:, :] = 0

            # transform
            fixed = translate(moving, [x_shift, y_shift], interpolation='bilinear', fill_mode='constant')
            # fixed = moving # debug

            size = moving.shape[0]
            x = max(0, int(x_shift+1))
            x_bound = min(size-s, size-s-x)         # exclude black edges
            while True:
                y = max(0, int(y_shift+1))
                y_bound = min(size-s, size-s-y)
                while True:
                    rand_choice_stride = random.randint(0, 15)
                    xx = min(x + rand_choice_stride * s // 16, x_bound)
                    yy = min(y + rand_choice_stride * s // 16, y_bound)
                    if yy != y_bound and xx != x_bound:
                        move = moving[xx:xx + s, yy:yy + s]
                        fix = fixed[xx:xx + s, yy:yy + s]
                        # yield move.astype(np.float32), fix.astype(np.float32), flow_gt
                        yield move.astype(np.float32), fix, flow_gt, loss_mask

                    if yy == y_bound:
                        break
                    y += stride
                if xx == x_bound:
                    break
                x += stride

    def augment(self, moving, fixed, flow, loss_mask):
        return moving, fixed, flow, loss_mask


class PairedImageRegistrationBatchLoader(BatchLoader):
    """Used for paired registration training (without DVF ground truth)"""
    def __init__(self, train_images, tc, num_slices, **kwargs):
        super(PairedImageRegistrationBatchLoader, self).__init__(train_images, tc, num_slices, **kwargs)

        if not hasattr(self.config, 'filter_blank'):
            self.config.filter_blank = False
        elif self.config.filter_blank:
            assert self.config.filter_threshold is not None

        # self.cur_filter_count = 0
        self.case_trial_limit = 10

    # def create_dataset_from_generator(self):
    #     return tf.data.Dataset.from_generator(self.parse_and_generate,
    #                                           output_types=(tf.float32, tf.float32),
    #                                           output_shapes=(tf.TensorShape([self.image_size, self.image_size, 3]),
    #                                                          tf.TensorShape([self.image_size, self.image_size, 3])))

    def create_dataset_from_generator(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.PATHS)
        return dataset.interleave(
            lambda x: tf.data.Dataset.from_generator(self.parse_and_generate, output_types=(tf.float32, tf.float32),
                                                     args=(x,),
                                                     output_shapes=(tf.TensorShape([self.image_size, self.image_size, self.input_channels]),
                                                                    tf.TensorShape([self.image_size, self.image_size, self.label_channels]))),
            cycle_length=self.config.n_threads, block_length=1, num_parallel_calls=self.config.n_threads)

    def parse_and_generate(self, path):
        s = self.config.image_size
        stride = s

        path = path.decode('UTF-8')

        if self.config.is_mat:
            fixed = loadmat(path)['output']
            moving = loadmat(path)['label']
        else:
            raise NotImplementedError()

        size = moving.shape[0]

        cur_trial_count = 0
        x = 0
        while True:
            y = 0
            while True:
                rand_choice_stride = random.randint(0, 15)
                xx = min(x + rand_choice_stride * s // 16, size - s)
                yy = min(y + rand_choice_stride * s // 16, size - s)
                if yy != size - s and xx != size - s:
                    fix = fixed[xx:xx + s, yy:yy + s]
                    mov = moving[xx:xx + s, yy:yy + s]

                    if self.config.filter_blank and np.mean(mov) >= self.config.filter_threshold \
                            and cur_trial_count < self.case_trial_limit:
                        cur_trial_count += 1
                        continue
                    else:
                        yield fix.astype(np.float32), mov.astype(np.float32)

                if yy == size - s:
                    break
                y += stride
            if xx == size - s:
                break
            x += stride

    def augment(self, fix, mov):
        fixmov = tf.concat([fix, mov], axis=-1)
        fixmov = tf.image.random_flip_left_right(fixmov)

        fix = fixmov[:, :, 0:self.config.num_slices]
        mov = fixmov[:, :, self.config.num_slices:self.config.num_slices+self.label_channels]

        random_number = tf.random.uniform(shape=(), minval=0, maxval=4, dtype=tf.int32)
        fix = tf.image.rot90(fix, k=random_number)
        mov = tf.image.rot90(mov, k=random_number)

        return fix, mov


class AffineImageRegistrationBatchLoader(BatchLoader):
    """Used for self-supervised registration pretraining"""
    def create_dataset_from_generator(self):
        return tf.data.Dataset.from_generator(self.parse_and_generate,
                                              output_types=(tf.float32, tf.float32),
                                              output_shapes=(tf.TensorShape([self.image_size, self.image_size, 3]),
                                                             tf.TensorShape([self.image_size, self.image_size, 3])))

    def parse_and_generate(self, path):
        s = self.config.image_size
        stride = s
        for path in self.PATHS:
            # print(path)
            if self.config.is_mat:
                moving = loadmat(path)['target'].astype(np.float32)
            else:
                moving = np.load(path).astype(np.float32)

            # generate a random flow
            maxshift = 40
            x_shift = int(random.uniform(-maxshift, maxshift))
            y_shift = int(random.uniform(-maxshift, maxshift))

            maxsheardegree = 5
            shear_degree = random.uniform(-maxsheardegree, maxsheardegree)

            # transform
            # Some problem with tx and ty... For now, use translate to perform shifting
            fixed = translate(moving, [x_shift, y_shift], interpolation='bilinear', fill_mode='constant').numpy()
            # if random.uniform(-1, 1) > 0:   # only half cases have shearing
            fixed = tf.keras.preprocessing.image.apply_affine_transform(fixed,
                                                                        theta=0, tx=0, ty=0, shear=shear_degree,
                                                                        fill_mode='nearest')

            size = moving.shape[0]
            x = max(0, int(x_shift+1))
            x_bound = min(size-s, size-s-x)         # exclude black edges
            while True:
                y = max(0, int(y_shift+1))
                y_bound = min(size-s, size-s-y)
                while True:
                    rand_choice_stride = random.randint(0, 15)
                    xx = min(x + rand_choice_stride * s // 16, x_bound)
                    yy = min(y + rand_choice_stride * s // 16, y_bound)
                    if yy != y_bound and xx != x_bound:
                        move = moving[xx:xx + s, yy:yy + s]
                        fix = fixed[xx:xx + s, yy:yy + s]

                        # check again if black edge is present in fix
                        if not np.all(fix):
                            continue

                        yield move.astype(np.float32), fix

                    if yy == y_bound:
                        break
                    y += stride
                if xx == x_bound:
                    break
                x += stride

    def augment(self, moving, fixed):
        imglab = tf.concat([moving, fixed], axis=-1)
        imglab = tf.image.random_flip_left_right(imglab)

        moving = imglab[:, :, 0:3]
        fixed = imglab[:, :, 3:6]

        random_number = tf.random.uniform(shape=(), minval=0, maxval=4, dtype=tf.int32)
        moving = tf.image.rot90(moving, k=random_number)
        fixed = tf.image.rot90(fixed, k=random_number)

        return moving, fixed

def Her2data_splitter(config):

    train_images = set(glob.glob(config.image_dir + '/*.mat'))
    valid_cases = config.valid_cases
    test_cases = config.test_cases

    valid_images, test_images = [], []

    for valid_case in valid_cases:
        valid_images = valid_images + glob.glob(config.image_dir + '/*' + valid_case + '*.mat')

    for test_case in test_cases:
        test_images = test_images + glob.glob(config.image_dir + '/*' + test_case + '*.mat')

    train_images = list(train_images - set(valid_images) - set(test_images))

    return train_images, valid_images, test_images
