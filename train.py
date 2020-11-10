from __future__ import absolute_import, division, print_function

import pickle

import numpy as np
import tensorflow as tf

import config
from resnet import resnet_18, resnet_18_fast3d, resnet_18_fast3d_split


def get_model(model_provider):
    model = model_provider()
    model.build(input_shape=(None, config.num_of_frames, config.image_height, config.image_width, config.channels))
    model.summary()
    return model


def batch_generator(x, y, batch_size=config.BATCH_SIZE):
    indices = np.arange(len(x))
    batch = []
    while True:
        np.random.shuffle(indices)
        for i in indices:
            batch.append(i)
            if len(batch) == batch_size:
                x_batch = x[batch]
                x_batch = x_batch / 255.0
                x_batch = x_batch.astype(np.float32)
                y_batch = y[batch]
                yield x_batch, y_batch
                batch = []


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')

    x_train = np.load('dataset/x_train2.npy')
    y_train = np.load('dataset/y_train2.npy')

    model = get_model(resnet_18_fast3d_split)


    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='weights/basic_2/resnet18_fast3d_split_{epoch:02d}-{categorical_accuracy:.4f}.h5',
        save_weights_only=True,
        monitor='categorical_accuracy',
        mode='max',
        save_best_only=True)

    model.compile(tf.keras.optimizers.Adam(learning_rate=0.001), tf.keras.losses.categorical_crossentropy,
                  tf.keras.metrics.categorical_accuracy)
    history = model.fit(batch_generator(x_train, y_train), epochs=30,
                        steps_per_epoch=x_train.shape[0] // config.BATCH_SIZE,
                        callbacks=[model_checkpoint_callback])

    with open('training/resnet18_fast3d_split.history', 'wb') as history_file:
        pickle.dump(history.history, history_file)
