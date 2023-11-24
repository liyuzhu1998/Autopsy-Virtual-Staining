import json
import numpy as np
import tensorflow as tf
import shutil, glob, sys


def copy_code(model_path):
    tf.io.gfile.mkdir(model_path + 'code')
    file_names = glob.glob('*.py')
    for file_name in file_names:
        try:
            shutil.copy(file_name, model_path + 'code')
        except:
            print("Failed to copy file: ", sys.exc_info())

    try:
        shutil.copytree('models/', model_path + 'code/models/')
    except:
        print('Skipped copying code in models/ because the source dir does not exist or the target dir already exists.')

    try:
        shutil.copytree('helper_scripts/', model_path + 'code/helper_scripts/')
    except:
        print('Skipped copying code in helper_scripts/ because the source dir does not exist or the target dir already exists.')


def verbose_msg(text, value, json_format=False):
    assert len(text) == len(value)

    if json_format:
        msg = json.dumps(json.loads(json.dumps({t: float(v) for (t, v) in zip(text, value)}),
                                    parse_float=lambda x: round(float(x), 4)))
    else:
        msg = ""
        for txt, v in zip(text, value):
            msg += str(txt) + ": {0:.4} || ".format(float(v))

    return msg


def print_and_save_msg(msg, file):
    print(msg)
    with open(file, 'a') as f:
        f.write(msg)


def freeze_model(model, freeze_batch_norm=False):
    '''
    freeze a keras model_G

    Input
    ----------
        model_G: a keras model_G
        freeze_batch_norm: False for not freezing batch normalization layers
    '''
    if freeze_batch_norm:
        for layer in model.layers:
            layer.trainable = False
    else:
        from tensorflow.keras.layers import BatchNormalization
        for layer in model.layers:
            if isinstance(layer, BatchNormalization):
                layer.trainable = True
            else:
                layer.trainable = False
    return model


def normalize(x):
    return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))
