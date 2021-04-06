import tensorflow as tf


def srvc_base(lr, scale, F, block_h, block_w):
    est = lr
    patches = tf.space_to_batch_nd(lr, block_shape=[block_h, block_w], paddings=[[0, 0], [0, 0]])

    features = tf.layers.conv2d(patches, 256, 3, strides=(1, 1), padding='valid',
                                data_format='channels_last', dilation_rate=(1, 1), activation=tf.nn.relu,
                                use_bias=True)
    kernel = tf.layers.conv2d(features, 3 * 3 * 3 * F, 3, strides=(1, 1), padding='valid',
                              data_format='channels_last', dilation_rate=(1, 1), activation=None,
                              use_bias=True)
    bias = tf.layers.conv2d(features, F, 3, strides=(1, 1), padding='valid',
                            data_format='channels_last', dilation_rate=(1, 1), activation=None,
                            use_bias=True)
    kernel = tf.reshape(kernel, [-1, 1, 1, 3 * 3 * 3, F])
    bias = tf.reshape(bias, [-1, 1, 1, F])

    patches = tf.image.extract_patches(patches, sizes=[1, 3, 3, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1],
                                       padding='SAME')
    patches = tf.expand_dims(patches, axis=3)
    patches = tf.matmul(patches, kernel)
    patches = tf.squeeze(patches, axis=3) + bias
    patches = tf.nn.relu(patches)
    est = tf.batch_to_space_nd(patches, block_shape=[block_h, block_w], crops=[[0, 0], [0, 0]])

    est = tf.layers.conv2d(est, 128, 5, strides=(1, 1), padding='same',
                           data_format='channels_last', dilation_rate=(1, 1), activation=tf.nn.relu,
                           use_bias=True)
    est = tf.layers.conv2d(est, 32, 3, strides=(1, 1), padding='same',
                           data_format='channels_last', dilation_rate=(1, 1), activation=tf.nn.relu,
                           use_bias=True)
    est = tf.layers.conv2d(est, 3 * scale * scale, 3, strides=(1, 1), padding='same',
                           data_format='channels_last', dilation_rate=(1, 1), activation=None,
                           use_bias=True)
    est = tf.nn.depth_to_space(est, scale, data_format='NHWC')
    indepth_est = est
    return indepth_est


def srvc(lr):
    scale = 4
    F = 32
    block_h = tf.shape(lr)[1] / 5
    block_w = tf.shape(lr)[2] / 5
    return srvc_base(lr, scale, F, block_h, block_w)
