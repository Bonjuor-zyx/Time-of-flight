import tensorflow as tf
import numpy as np

def inference(input, training=True):
    y1 = tf.layers.conv2d(input, filters=16, kernel_size=(3, 3), strides=(1, 1),
                          name='conv1', reuse=tf.AUTO_REUSE, padding='same')  # shape: 256, 256, 16

    y1_bn = tf.layers.batch_normalization(y1, axis=-1, name='bn_conv1',
                                       training=training, reuse=tf.AUTO_REUSE)
    y1_relu = tf.nn.relu(y1_bn)
    # first down sample
    y2 = tf.layers.conv2d(y1_relu, filters=16, kernel_size=(3, 3), strides=(2, 2),
                          name='conv2', reuse=tf.AUTO_REUSE, padding='same')  # shape: 128, 128, 16
    y2_bn = tf.layers.batch_normalization(y2, axis=-1, name='bn_conv2',
                                          training=training, reuse=tf.AUTO_REUSE)
    y2_relu = tf.nn.relu(y2_bn)
    # 再叠一层
    y3 = tf.layers.conv2d(y2_relu, filters=16, kernel_size=(3, 3), strides=(1, 1),
                          name='conv3', reuse=tf.AUTO_REUSE, padding='same')  # shape: 128, 128, 16
    y3_bn = tf.layers.batch_normalization(y3, axis=-1, name='bn_conv3',
                                          training=training, reuse=tf.AUTO_REUSE)
    y3_relu = tf.nn.relu(y3_bn)

    # second down sample
    y4 = tf.layers.conv2d(y3_relu, filters=16, kernel_size=(3, 3), strides=(2, 2),
                          name='conv4', reuse=tf.AUTO_REUSE, padding='same')  # shape: 64, 64, 16
    y4_bn = tf.layers.batch_normalization(y4, axis=-1, name='bn_conv4',
                                          training=training, reuse=tf.AUTO_REUSE)
    y4_relu = tf.nn.relu(y4_bn)

    # 再叠一层
    y5 = tf.layers.conv2d(y4_relu, filters=16, kernel_size=(3, 3), strides=(1, 1),
                          name='conv5', reuse=tf.AUTO_REUSE, padding='same')  # shape: 64, 64, 16
    y5_bn = tf.layers.batch_normalization(y5, axis=-1, name='bn_conv5',
                                          training=training, reuse=tf.AUTO_REUSE)
    y5_relu = tf.nn.relu(y5_bn)

    # third down sample
    y6 = tf.layers.conv2d(y5_relu, filters=32, kernel_size=(3, 3), strides=(2, 2),
                          name='conv6', reuse=tf.AUTO_REUSE, padding='same')  # shape: 32, 32, 32
    y6_bn = tf.layers.batch_normalization(y6, axis=-1, name='bn_conv6',
                                          training=training, reuse=tf.AUTO_REUSE)
    y6_relu = tf.nn.relu(y6_bn)

    # 再叠一层
    y7 = tf.layers.conv2d(y6_relu, filters=32, kernel_size=(3, 3), strides=(1, 1),
                          name='conv7', reuse=tf.AUTO_REUSE, padding='same')  # shape:32, 32, 32
    y7_bn = tf.layers.batch_normalization(y7, axis=-1, name='bn_conv7',
                                          training=training, reuse=tf.AUTO_REUSE)
    y7_relu = tf.nn.relu(y7_bn)

    # fourth down sample
    y8 = tf.layers.conv2d(y7_relu, filters=32, kernel_size=(3, 3), strides=(2, 2),
                          name='conv8', reuse=tf.AUTO_REUSE, padding='same')  # shape: 16, 16, 32
    y8_bn = tf.layers.batch_normalization(y8, axis=-1, name='bn_conv8',
                                          training=training, reuse=tf.AUTO_REUSE)
    y8_relu = tf.nn.relu(y8_bn)

    # 再叠一层
    y9 = tf.layers.conv2d(y8_relu, filters=32, kernel_size=(3, 3), strides=(1, 1),
                          name='conv9', reuse=tf.AUTO_REUSE, padding='same')  # shape: 16, 16, 32
    y9_bn = tf.layers.batch_normalization(y9, axis=-1, name='bn_conv9',
                                          training=training, reuse=tf.AUTO_REUSE)
    y9_relu = tf.nn.relu(y9_bn)

    # fifth down sample
    y10 = tf.layers.conv2d(y9_relu, filters=32, kernel_size=(3, 3), strides=(2, 2),
                           name='conv10', reuse=tf.AUTO_REUSE, padding='same')  # shape: 8, 8, 32
    y10_bn = tf.layers.batch_normalization(y10, axis=-1, name='bn_conv10',
                                           training=training, reuse=tf.AUTO_REUSE)
    y10_relu = tf.nn.relu(y10_bn)  # shape: 8, 8, 32, 开始上采样
    print(y10_relu.shape)

    # 上采样开始
    # first up sample
    dy1 = tf.layers.conv2d_transpose(y10_relu, filters=32, kernel_size=(3, 3), strides=(2, 2),
                                     name='deconv1', reuse=tf.AUTO_REUSE, padding='same')  # shape: 16, 16, 32
    dy1_bn = tf.layers.batch_normalization(dy1, axis=-1, name='bn_deconv1',
                                           training=training, reuse=tf.AUTO_REUSE)
    add_dy1 = tf.add(y9, dy1_bn)
    dy1_relu = tf.nn.relu(add_dy1)
    # second up sample
    dy2 = tf.layers.conv2d_transpose(dy1_relu, filters=32, kernel_size=(3, 3), strides=(2, 2),
                                     name='deconv2', reuse=tf.AUTO_REUSE, padding='same')  # shape: 32, 32, 32
    dy2_bn = tf.layers.batch_normalization(dy2, axis=-1, name='bn_deconv2',
                                           training=training, reuse=tf.AUTO_REUSE)
    add_dy2 = tf.add(y7, dy2_bn)
    dy2_relu = tf.nn.relu(add_dy2)
    #third up sample
    dy3 = tf.layers.conv2d_transpose(dy2_relu, filters=16, kernel_size=(3, 3), strides=(2, 2),
                                     name='deconv3', reuse=tf.AUTO_REUSE, padding='same')  # shape: 64, 64, 16
    dy3_bn = tf.layers.batch_normalization(dy3, axis=-1, name='bn_deconv3',
                                           training=training, reuse=tf.AUTO_REUSE)
    add_dy3 = tf.add(dy3, dy3_bn)
    dy3_relu = tf.nn.relu(add_dy3)
    #fourth up sample
    dy4 = tf.layers.conv2d_transpose(dy3_relu, filters=16, kernel_size=(3, 3), strides=(2, 2),
                                     name='deconv4', reuse=tf.AUTO_REUSE, padding='same')  # shape: 128, 128, 16
    dy4_bn = tf.layers.batch_normalization(dy4, axis=-1, name='bn_deconv4',
                                           training=training, reuse=tf.AUTO_REUSE)
    add_dy4 = tf.add(y3, dy4_bn)
    dy4_relu = tf.nn.relu(add_dy4)
    #fifth up sample
    dy5 = tf.layers.conv2d_transpose(dy4_relu, filters=16, kernel_size=(3, 3), strides=(2, 2),
                                     name='deconv5', reuse=tf.AUTO_REUSE, padding='same')  # shape: 256, 256, 16
    dy5_bn = tf.layers.batch_normalization(dy5, axis=-1, name='bn_deconv5',
                                           training=training, reuse=tf.AUTO_REUSE)
    add_dy5 = tf.add(y1, dy5_bn)
    dy5_relu = tf.nn.relu(add_dy5)

    # 得到深度图

    predict_depth = tf.layers.conv2d(dy5_relu, filters=1, kernel_size=(3, 3), strides=(1, 1),
                                     name='final', reuse=tf.AUTO_REUSE, padding='same')  # shape: 256, 256, 1
    predict_depth = tf.nn.relu(predict_depth)

    return predict_depth



