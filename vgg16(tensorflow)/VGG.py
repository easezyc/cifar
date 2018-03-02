import tensorflow as tf
import random

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(input, k_size=1, stride=1, name=None):
    return tf.nn.max_pool(input, ksize=[1, k_size, k_size, 1], strides=[1, stride, stride, 1], padding='SAME',
                          name=name)


def batch_norm(input, train_flag):
    return tf.contrib.layers.batch_norm(input, decay=0.9, center=True, scale=True, epsilon=1e-3, is_training=train_flag,
                                        updates_collections=None)


def vgg19(input, keep_prob, train_flag, class_num, onemore_layer, stop_flag, is_relu, seed):
    random.seed(seed)
    l_random = []
    for i in range(20):
        l_random.append(random.randint(1,100))
    print(l_random)
    l_restore = []
    W_conv1_1 = tf.get_variable('conv1_1', shape=[3, 3, 3, 64], initializer=tf.contrib.keras.initializers.he_normal(seed=l_random[0]))
    b_conv1_1 = bias_variable([64])
    output = tf.nn.relu(batch_norm(conv2d(input, W_conv1_1) + b_conv1_1, train_flag))

    W_conv1_2 = tf.get_variable('conv1_2', shape=[3, 3, 64, 64], initializer=tf.contrib.keras.initializers.he_normal(seed=l_random[1]))
    b_conv1_2 = bias_variable([64])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv1_2) + b_conv1_2, train_flag))
    output = max_pool(output, 2, 2, "pool1")

    W_conv2_1 = tf.get_variable('conv2_1', shape=[3, 3, 64, 128], initializer=tf.contrib.keras.initializers.he_normal(seed=l_random[2]))
    b_conv2_1 = bias_variable([128])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv2_1) + b_conv2_1, train_flag))

    W_conv2_2 = tf.get_variable('conv2_2', shape=[3, 3, 128, 128],
                                initializer=tf.contrib.keras.initializers.he_normal(seed=l_random[3]))
    b_conv2_2 = bias_variable([128])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv2_2) + b_conv2_2, train_flag))
    output = max_pool(output, 2, 2, "pool2")

    W_conv3_1 = tf.get_variable('conv3_1', shape=[3, 3, 128, 256],
                                initializer=tf.contrib.keras.initializers.he_normal(seed=l_random[4]))
    b_conv3_1 = bias_variable([256])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv3_1) + b_conv3_1, train_flag))

    W_conv3_2 = tf.get_variable('conv3_2', shape=[3, 3, 256, 256],
                                initializer=tf.contrib.keras.initializers.he_normal(seed=l_random[5]))
    b_conv3_2 = bias_variable([256])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv3_2) + b_conv3_2, train_flag))

    W_conv3_3 = tf.get_variable('conv3_3', shape=[3, 3, 256, 256],
                                initializer=tf.contrib.keras.initializers.he_normal(seed=l_random[6]))
    b_conv3_3 = bias_variable([256])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv3_3) + b_conv3_3, train_flag))

    W_conv3_4 = tf.get_variable('conv3_4', shape=[3, 3, 256, 256],
                                initializer=tf.contrib.keras.initializers.he_normal(seed=l_random[7]))
    b_conv3_4 = bias_variable([256])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv3_4) + b_conv3_4, train_flag))
    output = max_pool(output, 2, 2, "pool3")

    W_conv4_1 = tf.get_variable('conv4_1', shape=[3, 3, 256, 512],
                                initializer=tf.contrib.keras.initializers.he_normal(seed=l_random[8]))
    b_conv4_1 = bias_variable([512])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv4_1) + b_conv4_1, train_flag))

    W_conv4_2 = tf.get_variable('conv4_2', shape=[3, 3, 512, 512],
                                initializer=tf.contrib.keras.initializers.he_normal(seed=l_random[9]))
    b_conv4_2 = bias_variable([512])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv4_2) + b_conv4_2, train_flag))

    W_conv4_3 = tf.get_variable('conv4_3', shape=[3, 3, 512, 512],
                                initializer=tf.contrib.keras.initializers.he_normal(seed=l_random[10]))
    b_conv4_3 = bias_variable([512])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv4_3) + b_conv4_3, train_flag))

    W_conv4_4 = tf.get_variable('conv4_4', shape=[3, 3, 512, 512],
                                initializer=tf.contrib.keras.initializers.he_normal(seed=l_random[11]))
    b_conv4_4 = bias_variable([512])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv4_4), train_flag) + b_conv4_4)
    output = max_pool(output, 2, 2)

    W_conv5_1 = tf.get_variable('conv5_1', shape=[3, 3, 512, 512],
                                initializer=tf.contrib.keras.initializers.he_normal(seed=l_random[12]))
    b_conv5_1 = bias_variable([512])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv5_1) + b_conv5_1, train_flag))

    W_conv5_2 = tf.get_variable('conv5_2', shape=[3, 3, 512, 512],
                                initializer=tf.contrib.keras.initializers.he_normal(seed=l_random[13]))
    b_conv5_2 = bias_variable([512])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv5_2) + b_conv5_2, train_flag))

    W_conv5_3 = tf.get_variable('conv5_3', shape=[3, 3, 512, 512],
                                initializer=tf.contrib.keras.initializers.he_normal(seed=l_random[14]))
    b_conv5_3 = bias_variable([512])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv5_3) + b_conv5_3, train_flag))

    W_conv5_4 = tf.get_variable('conv5_4', shape=[3, 3, 512, 512],
                                initializer=tf.contrib.keras.initializers.he_normal(seed=l_random[15]))
    b_conv5_4 = bias_variable([512])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv5_4) + b_conv5_4, train_flag))

    # output = tf.contrib.layers.flatten(output)
    output = tf.reshape(output, [-1, 2 * 2 * 512])

    W_fc1 = tf.get_variable('fc1', shape=[2048, 4096], initializer=tf.contrib.keras.initializers.he_normal(seed=l_random[16]))
    b_fc1 = bias_variable([4096])
    output = tf.nn.relu(batch_norm(tf.matmul(output, W_fc1) + b_fc1, train_flag))
    output = tf.nn.dropout(output, keep_prob)

    W_fc2 = tf.get_variable('fc2', shape=[4096, 4096], initializer=tf.contrib.keras.initializers.he_normal(seed=l_random[17]))
    b_fc2 = bias_variable([4096])
    output = tf.nn.relu(batch_norm(tf.matmul(output, W_fc2) + b_fc2, train_flag))
    output = tf.nn.dropout(output, keep_prob)

    if onemore_layer:
        W_fc3 = tf.get_variable('fc3', shape=[4096, 300], initializer=tf.contrib.keras.initializers.he_normal(seed=l_random[18]))
        b_fc3 = bias_variable([300])
        l_restore.append(W_fc3)
        l_restore.append(b_fc3)
        output = tf.nn.relu(batch_norm(tf.matmul(output, W_fc3) + b_fc3, train_flag))
        if stop_flag:
            output = tf.stop_gradient(output)

        W_fc4 = tf.get_variable('fc4', shape=[300, class_num], initializer=tf.contrib.keras.initializers.he_normal(seed=l_random[19]))
        b_fc4 = bias_variable([class_num])
        if is_relu:
            output = tf.nn.relu(batch_norm(tf.matmul(output, W_fc4) + b_fc4, train_flag))
        else:
            output = batch_norm(tf.matmul(output, W_fc4) + b_fc4, train_flag)
    else:
        if stop_flag:
            output = tf.stop_gradient(output)
        W_fc3 = tf.get_variable('fc3', shape=[4096, class_num], initializer=tf.contrib.keras.initializers.he_normal(seed=l_random[18]))
        b_fc3 = bias_variable([class_num])
        print(b_fc3)
        if is_relu:
            output = tf.nn.relu(batch_norm(tf.matmul(output, W_fc3) + b_fc3, train_flag))
        else:
            output = batch_norm(tf.matmul(output, W_fc3) + b_fc3, train_flag)

    return output
