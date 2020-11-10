import tensorflow as tf
from config import NUM_CLASSES
from residual_block import make_basic_block3d_layer, make_fast_conv3d_block_layer, make_fast_conv3d_split_block_layer


class ResNetBasic(tf.keras.Model):
    def __init__(self, layer_params):
        super(ResNetBasic, self).__init__()

        self.conv1 = tf.keras.layers.Conv3D(filters=64,
                                            kernel_size=(7, 7, 7),
                                            strides=2,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool3D(pool_size=(3, 3, 3),
                                               strides=2,
                                               padding="same")

        self.layer1 = make_basic_block3d_layer(filter_num=64,
                                               blocks=layer_params[0])
        self.layer2 = make_basic_block3d_layer(filter_num=128,
                                               blocks=layer_params[1],
                                               stride=2)
        self.layer3 = make_basic_block3d_layer(filter_num=256,
                                               blocks=layer_params[2],
                                               stride=2)
        self.layer4 = make_basic_block3d_layer(filter_num=512,
                                               blocks=layer_params[3],
                                               stride=2)

        self.avgpool = tf.keras.layers.GlobalAveragePooling3D()
        self.fc = tf.keras.layers.Dense(units=NUM_CLASSES, activation=tf.keras.activations.softmax)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.avgpool(x)
        output = self.fc(x)

        return output


class ResNetTypeFast3d(tf.keras.Model):
    def __init__(self, layer_params):
        super(ResNetTypeFast3d, self).__init__()

        self.conv1 = tf.keras.layers.Conv3D(filters=64,
                                            kernel_size=(7, 7, 7),
                                            strides=2,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool3D(pool_size=(3, 3, 3),
                                               strides=2,
                                               padding="same")

        self.layer1 = make_fast_conv3d_block_layer(filter_num=64,
                                                   blocks=layer_params[0])
        self.layer2 = make_fast_conv3d_block_layer(filter_num=128,
                                                   blocks=layer_params[1],
                                                   stride=(1, 2, 2))
        self.layer3 = make_fast_conv3d_block_layer(filter_num=256,
                                                   blocks=layer_params[2],
                                                   stride=(1, 2, 2))
        self.layer4 = make_fast_conv3d_block_layer(filter_num=512,
                                                   blocks=layer_params[3],
                                                   stride=2)

        self.avgpool = tf.keras.layers.GlobalAveragePooling3D()
        self.fc = tf.keras.layers.Dense(units=NUM_CLASSES, activation=tf.keras.activations.softmax)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.avgpool(x)
        output = self.fc(x)

        return output


class ResNetTypeFast3dSplit(tf.keras.Model):
    def __init__(self, layer_params):
        super(ResNetTypeFast3dSplit, self).__init__()

        self.conv1 = tf.keras.layers.Conv3D(filters=64,
                                            kernel_size=(7, 7, 7),
                                            strides=2,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool3D(pool_size=(3, 3, 3),
                                               strides=2,
                                               padding="same")

        self.layer1 = make_fast_conv3d_split_block_layer(filter_num=64,
                                                         blocks=layer_params[0])
        self.layer2 = make_fast_conv3d_split_block_layer(filter_num=128,
                                                         blocks=layer_params[1],
                                                         stride=(1, 2, 2))
        self.layer3 = make_fast_conv3d_split_block_layer(filter_num=256,
                                                         blocks=layer_params[2],
                                                         stride=(1, 2, 2))
        self.layer4 = make_fast_conv3d_split_block_layer(filter_num=512,
                                                         blocks=layer_params[3],
                                                         stride=2)

        self.avgpool = tf.keras.layers.GlobalAveragePooling3D()
        self.fc = tf.keras.layers.Dense(units=NUM_CLASSES, activation=tf.keras.activations.softmax)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.avgpool(x)
        output = self.fc(x)

        return output


def resnet_18():
    return ResNetBasic(layer_params=[2, 2, 2, 2])


def resnet_34():
    return ResNetBasic(layer_params=[3, 4, 6, 3])


def resnet_18_fast3d():
    return ResNetTypeFast3d(layer_params=[2, 2, 2, 2])


def resnet_34_fast3d():
    return ResNetTypeFast3d(layer_params=[3, 4, 6, 3])


def resnet_18_fast3d_split():
    return ResNetTypeFast3dSplit(layer_params=[2, 2, 2, 2])


def resnet_34_fast3d_split():
    return ResNetTypeFast3dSplit(layer_params=[3, 4, 6, 3])
