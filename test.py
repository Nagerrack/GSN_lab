# import numpy as np
#
# a = np.array([[1, 2], [3, 4]])
#
# print(a)
# print(a.shape)
# print(np.repeat(a[:, :, np.newaxis], 3, axis=2))
# print(np.repeat(a[np.newaxis, :, :], 3, axis=0))
# print(np.repeat(a[np.newaxis, :, :], 3, axis=0).shape)


from __future__ import absolute_import, division, print_function

import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import config
from resnet import resnet_18, resnet_18_fast3d, resnet_18_fast3d_split


def get_model(model_provider):
    model = model_provider()
    model.build(input_shape=(None, config.num_of_frames, config.image_height, config.image_width, config.channels))
    model.summary()
    return model


def test_batch_generator(x, batch_size=config.BATCH_SIZE):
    indices = np.arange(len(x))
    batch = []
    for i in indices:
        batch.append(i)
        if len(batch) == batch_size:
            x_batch = x[batch]
            x_batch = x_batch / 255.0
            x_batch = x_batch.astype(np.float32)
            yield x_batch


def split_into_batches(x, batch_size=config.BATCH_SIZE):
    indices = np.arange(len(x))
    i = 0
    while i < len(x):
        b = x[indices[i:min(i + batch_size, len(x))]]
        b = b / 255.0
        b = b.astype(np.float32)
        yield b
        i += batch_size


def get_validation_losses_and_acc_for_epochs(model, x, y, weight_path):
    epochs = []
    top1_accs = []
    top5_accs = []
    losses = []
    print(weight_path)
    for w_file_name in os.listdir('./' + weight_path):
        epoch_num = int(w_file_name.split('_')[-1].split('-')[0])
        model.load_weights(weight_path + '/' + w_file_name)
        y_preds = []
        for batch in tqdm(split_into_batches(x)):
            y_pred = model.predict(batch)
            y_preds.append(y_pred)
        y_preds = np.vstack(y_preds)

        top1_acc = np.mean(tf.keras.metrics.categorical_accuracy(y, y_preds))
        top5_acc = np.mean(tf.keras.metrics.top_k_categorical_accuracy(y, y_preds))
        loss = np.mean(tf.keras.losses.categorical_crossentropy(y, y_preds))
        epochs.append(epoch_num)
        top1_accs.append(top1_acc)
        top5_accs.append(top5_acc)
        losses.append(loss)

    print((epochs, top1_accs, top5_accs, losses))
    return epochs, top1_accs, top5_accs, losses


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')

    x_test = np.load('dataset/x_test2.npy')

    y_test = np.load('dataset/y_test2.npy')

    # model.load_weights('weights/basic_2/resnet18_3d_20-0.9738.h5')
    # y_preds = []
    # for batch in tqdm(split_into_batches(x_test)):
    #     y_pred = model.predict(batch)
    #     y_preds.append(y_pred)
    # y_preds = np.vstack(y_preds)
    #
    # top1_acc = np.mean(tf.keras.metrics.categorical_accuracy(y_test, y_preds))
    # top5_acc = np.mean(tf.keras.metrics.top_k_categorical_accuracy(y_test, y_preds))
    #
    # loss = np.mean(tf.keras.losses.categorical_crossentropy(y_test, y_preds))
    #
    # print(top1_acc)
    # print(top5_acc)

    # get_validation_losses_and_acc_for_epochs(model, x_test, y_test, 'weights/basic_2')

    model = get_model(resnet_18_fast3d)
    get_validation_losses_and_acc_for_epochs(model, x_test, y_test, 'weights/fast_1')

    model = get_model(resnet_18_fast3d_split)
    get_validation_losses_and_acc_for_epochs(model, x_test, y_test, 'weights/fast_split_1')
