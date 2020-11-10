import tensorflow as tf


class BasicBlock3d(tf.keras.layers.Layer):

    def __init__(self, filter_num, stride=1):
        super(BasicBlock3d, self).__init__()
        self.conv1 = tf.keras.layers.Conv3D(filters=filter_num,
                                            kernel_size=(3, 3, 3),
                                            strides=stride,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv3D(filters=filter_num,
                                            kernel_size=(3, 3, 3),
                                            strides=1,
                                            padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        if stride != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv3D(filters=filter_num,
                                                       kernel_size=(1, 1, 1),
                                                       strides=stride))
            self.downsample.add(tf.keras.layers.BatchNormalization())
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)

        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output


class FastConv3dBlock(tf.keras.layers.Layer):

    def __init__(self, filter_num, stride=1):
        super(FastConv3dBlock, self).__init__()
        self.conv1a = tf.keras.layers.Conv3D(filters=filter_num,
                                             kernel_size=(1, 3, 3),
                                             strides=stride,
                                             padding="same")
        self.bn1a = tf.keras.layers.BatchNormalization()
        self.conv1b = tf.keras.layers.Conv3D(filters=filter_num,
                                             kernel_size=(3, 3, 1),
                                             strides=1,
                                             padding="same")
        self.bn1b = tf.keras.layers.BatchNormalization()
        self.conv1c = tf.keras.layers.Conv3D(filters=filter_num,
                                             kernel_size=(3, 1, 3),
                                             strides=1,
                                             padding="same")
        self.bn1c = tf.keras.layers.BatchNormalization()

        self.conv2a = tf.keras.layers.Conv3D(filters=filter_num,
                                             kernel_size=(1, 3, 3),
                                             strides=1,
                                             padding="same")
        self.bn2a = tf.keras.layers.BatchNormalization()
        self.conv2b = tf.keras.layers.Conv3D(filters=filter_num,
                                             kernel_size=(3, 3, 1),
                                             strides=1,
                                             padding="same")
        self.bn2b = tf.keras.layers.BatchNormalization()
        self.conv2c = tf.keras.layers.Conv3D(filters=filter_num,
                                             kernel_size=(3, 1, 3),
                                             strides=1,
                                             padding="same")
        self.bn2c = tf.keras.layers.BatchNormalization()

        # (2, 28, 28, 128) (3, 56, 56, 128)
        if stride != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv3D(filters=filter_num,
                                                       kernel_size=(1, 1, 1),
                                                       strides=stride))
            self.downsample.add(tf.keras.layers.BatchNormalization())
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        x = self.conv1a(inputs)
        x = self.bn1a(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv1b(x)
        x = self.bn1b(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv1c(x)
        x = self.bn1c(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2a(x)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output


class FastConv3dSplitBlock(tf.keras.layers.Layer):

    def __init__(self, filter_num, stride=1):
        super(FastConv3dSplitBlock, self).__init__()
        self.conv1a = tf.keras.layers.Conv3D(filters=filter_num,
                                             kernel_size=(1, 3, 3),
                                             strides=stride,
                                             padding="same")
        self.bn1a = tf.keras.layers.BatchNormalization()
        self.conv1b = tf.keras.layers.Conv3D(filters=filter_num,
                                             kernel_size=(3, 3, 1),
                                             strides=1,
                                             padding="same")
        self.bn1b = tf.keras.layers.BatchNormalization()
        self.conv1c = tf.keras.layers.Conv3D(filters=filter_num,
                                             kernel_size=(3, 1, 3),
                                             strides=1,
                                             padding="same")
        self.bn1c = tf.keras.layers.BatchNormalization()

        self.conv2a = tf.keras.layers.Conv3D(filters=filter_num,
                                             kernel_size=(1, 3, 3),
                                             strides=1,
                                             padding="same")
        self.bn2a = tf.keras.layers.BatchNormalization()
        self.conv2b = tf.keras.layers.Conv3D(filters=filter_num,
                                             kernel_size=(3, 3, 1),
                                             strides=1,
                                             padding="same")
        self.bn2b = tf.keras.layers.BatchNormalization()
        self.conv2c = tf.keras.layers.Conv3D(filters=filter_num,
                                             kernel_size=(3, 1, 3),
                                             strides=1,
                                             padding="same")
        self.bn2c = tf.keras.layers.BatchNormalization()

        # (2, 28, 28, 128) (3, 56, 56, 128)
        if stride != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv3D(filters=filter_num,
                                                       kernel_size=(1, 1, 1),
                                                       strides=stride))
            self.downsample.add(tf.keras.layers.BatchNormalization())
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        x = self.conv1a(inputs)
        x = self.bn1a(x, training=training)
        x = tf.nn.relu(x)

        x_b = self.conv1b(x)
        x_b = self.bn1b(x_b, training=training)
        x_c = self.conv1c(x)
        x_c = self.bn1c(x_c, training=training)
        x = tf.nn.relu(tf.keras.layers.add([x_b, x_c]))

        x = self.conv2a(x)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x_b = self.conv2b(x)
        x_b = self.bn2b(x_b, training=training)
        x_c = self.conv2c(x)
        x_c = self.bn2c(x_c, training=training)
        x = tf.nn.relu(tf.keras.layers.add([x_b, x_c]))

        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output


def make_basic_block3d_layer(filter_num, blocks, stride=1):
    res_block = tf.keras.Sequential()
    res_block.add(BasicBlock3d(filter_num, stride=stride))

    for _ in range(1, blocks):
        res_block.add(BasicBlock3d(filter_num, stride=1))

    return res_block


def make_fast_conv3d_block_layer(filter_num, blocks, stride=1):
    res_block = tf.keras.Sequential()
    res_block.add(FastConv3dBlock(filter_num, stride=stride))

    for _ in range(1, blocks):
        res_block.add(FastConv3dBlock(filter_num, stride=1))

    return res_block


def make_fast_conv3d_split_block_layer(filter_num, blocks, stride=1):
    res_block = tf.keras.Sequential()
    res_block.add(FastConv3dSplitBlock(filter_num, stride=stride))

    for _ in range(1, blocks):
        res_block.add(FastConv3dSplitBlock(filter_num, stride=1))

    return res_block
