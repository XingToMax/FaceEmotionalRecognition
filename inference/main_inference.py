import tensorflow as tf

INPUT_NODE = 2304
OUTPUT_NODE = 7

IMAGE_SIZE = 48
NUM_CHANNELS = 3
NUM_LABELS = 10

KEEP_PROP = 0.3

KEEP_PROP_NEXT = 0.3

# 定义卷积网络的前向传播过程。
# input_tensor : 输入训练数据
# train : 标记为训练过程还是测试过程，若为训练过程，需要进行dropout
def inference(input_tensor, train):
    # 第一个卷积层，包括一个核大小5*5，步长为1，SAME模式的卷积核
    # 激励函数为relu
    # 输入为48 * 48 * 1
    # 输出为48 * 48 * 32
    with tf.variable_scope('layer1-conv1'):
        layer1_weight = tf.get_variable(
            "weight", [5,5,1,32], initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        layer1_bias = tf.get_variable(
            "bias", [32], initializer=tf.constant_initializer(0.1))
        layer1_conv = tf.nn.conv2d(
            input_tensor, layer1_weight, strides=[1,1,1,1], padding='SAME')
        layer1_relu = tf.nn.relu(tf.nn.bias_add(layer1_conv, layer1_bias))

    print('第一个卷积层输出size', layer1_relu.get_shape().as_list())

    # 第一个池化层，大小为3*3，步长为2, SAME
    # 输出为24 * 24 * 32
    with tf.variable_scope('layer2-pool1'):
        layer2_pool = tf.nn.max_pool(layer1_relu, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

    print('第一个池化层输出size', layer2_pool.get_shape().as_list())

    # 第二个卷积层
    # 激励函数为relu
    # 输出为24 * 24 * 32
    with tf.variable_scope('layer3-conv2'):
        layer3_weight = tf.get_variable(
            "weight", [4, 4, 32, 32], initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        layer3_bias = tf.get_variable(
            "bias", [32], initializer=tf.constant_initializer(0.1))
        layer3_conv = tf.nn.conv2d(
            layer2_pool, layer3_weight, strides=[1, 1, 1, 1], padding='SAME')
        layer3_relu = tf.nn.relu(tf.nn.bias_add(layer3_conv, layer3_bias))

    print('第二个卷积层输出size', layer3_relu.get_shape().as_list())

    # 第二个池化层
    # 输出为12 * 12 * 32
    with tf.variable_scope('layer4-pool2'):
        layer4_pool = tf.nn.max_pool(layer3_relu, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

    print('第二个池化层输出size', layer4_pool.get_shape().as_list())

    # 第三个卷积层
    # 激励函数为relu
    # 输出为12 * 12 * 64
    with tf.variable_scope('layer5-conv3'):
        layer5_weight = tf.get_variable(
            "weight", [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        layer5_bias = tf.get_variable(
            "bias", [64], initializer=tf.constant_initializer(0.1))
        layer5_conv = tf.nn.conv2d(
            layer4_pool, layer5_weight, strides=[1, 1, 1, 1], padding='SAME')
        layer5_relu = tf.nn.relu(tf.nn.bias_add(layer5_conv, layer5_bias))

    print('第三个卷积层输出size', layer5_relu.get_shape().as_list())

    # 第三个池化层
    # 输出为6 * 6 * 64
    with tf.variable_scope('layer6-pool3'):
        layer6_pool = tf.nn.max_pool(layer5_relu, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

    print('第三个池化层输出size', layer6_pool.get_shape().as_list())

    # 将各层展平，进入全连接层
    # 这里应该变为1 * 36 * 64
    # pool_shape = layer6_pool.get_shape().as_list()
    # nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    # reshaped = tf.reshape(layer6_pool, [pool_shape[0], nodes])
    reshaped = tf.reshape(layer6_pool, [-1, 2304])

    # 第一个全连接层
    # 输出为1 * 1 * 2048
    with tf.variable_scope('layer7-fc1'):
        layer7_weight = tf.get_variable(
            "weight", [2304, 2048], initializer=tf.truncated_normal_initializer(stddev=0.1))
        layer7_bias = tf.get_variable("bias", [2048], initializer=tf.constant_initializer(0.1))
        layer7_fc = tf.nn.relu(tf.matmul(reshaped, layer7_weight) + layer7_bias)
        if train :
            layer7_fc = tf.nn.dropout(layer7_fc, KEEP_PROP, name='layer7')

    # 第二个全连接层
    # 输出为1 * 1 * 1024
    with tf.variable_scope('layer8-fc2'):
        layer8_weight = tf.get_variable(
            "weight", [2048, 1024], initializer=tf.truncated_normal_initializer(stddev=0.1))
        layer8_bias = tf.get_variable("bias", [1024], initializer=tf.constant_initializer(0.1))
        layer8_fc = tf.nn.relu(tf.matmul(layer7_fc, layer8_weight) + layer8_bias)
        if train:
            layer8_fc = tf.nn.dropout(layer8_fc, KEEP_PROP_NEXT, name='dropout')

    # 第三个全连接层
    # 输出为1 * 1 * 7
    with tf.variable_scope('layer9-fc3'):
        layer9_weight = tf.get_variable(
            "weight", [1024, 7], initializer=tf.truncated_normal_initializer(stddev=0.1))
        layer9_bias = tf.get_variable("bias", [7], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(layer8_fc, layer9_weight, name='logits') + layer9_bias
    return logit





