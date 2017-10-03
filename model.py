import tensorflow as tf


VGG_MEAN = [103.939, 116.779, 123.68]


def constant_weight_variable(data_dict, name):
    init_val = data_dict[name][0]
    return tf.get_variable(
        name + "_weights",
        initializer=tf.constant(init_val, name="weights"))


def constant_bias_variable(data_dict, name):
    init_val = data_dict[name][1]
    return tf.get_variable(
        name + "_biases",
        initializer=tf.constant(init_val, name="biases"))


def weight_variable(shape, name="weights", var_init=0.1):
    """Wrap weight variable initialization of tensorflow."""
    return tf.get_variable(
        name, shape,
        initializer=tf.truncated_normal_initializer(stddev=var_init))


def bias_variable(shape, name="biases"):
    """Wrapp bias variable initialization of tensorflow."""
    return tf.get_variable(
        name, shape, initializer=tf.constant_initializer(0.1))


def conv2d(x, W):
    """Wrap convolution layer initialization of tensorflow."""
    return tf.nn.conv2d(
        x, W, strides=[1, 1, 1, 1], padding="SAME", name="convolution")


def max_pool_2x2(x):
    """Wrap the max-pool layer initialization of tensorflow."""
    ksize = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    return tf.nn.max_pool(
        x, ksize=ksize, strides=strides, padding="SAME", name="max-pooling")


def conv_layer(x, n_colors, n_kernels, name, reuse=None):
        """Create a convolutionnal layer."""
        with tf.variable_scope(name, reuse=reuse):
            filt = weight_variable(
                shape=[3, 3, n_colors, n_kernels], name=name + "_filters")

            conv = conv2d(x, filt)

            conv_biases = bias_variable(
                shape=[n_kernels], name=name + "_biases")
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu


def constant_conv_layer(x, data_dict, name, reuse=None):
        """Create a constant convolutionnal layer."""
        with tf.variable_scope(name, reuse=reuse):
            filt = constant_weight_variable(data_dict, name)

            conv = conv2d(x, filt)

            conv_biases = constant_bias_variable(data_dict, name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu


def fc_layer(x, in_size, fc_dim, name, reuse=None):
    """Create a fully-connected layer."""
    with tf.variable_scope(name, reuse=reuse):
        weights = weight_variable(
            [in_size, fc_dim], name=name + "_weights")
        biases = bias_variable([fc_dim], name=name + "_biases")

        x = tf.reshape(x, [-1, in_size])
        y = tf.nn.bias_add(tf.matmul(x, weights), biases)
    return y


def constant_fc_layer(x, data_dict, in_size, name, reuse=None):
    """Create a constant fully-connected layer."""
    with tf.variable_scope(name, reuse=reuse):
        weights = constant_weight_variable(data_dict, name)
        biases = constant_bias_variable(data_dict, name)

        x = tf.reshape(x, [-1, in_size])
        y = tf.nn.bias_add(tf.matmul(x, weights), biases)
    return y


def constant_convolutional_layers(
    input_var, data_dict, n_conv, series_id=0, reuse=None
):
    """Create a constant VGG like series of convolutional layers.

    This function creates a series of convolutional layers followed by a max
    pooling.

    Arguments:
        - input_var (tf tensor): the input of the convolutional layer.
        - data_dict (dict): the dictionnary containing the value for each
        variable.
        - n_conv (int): the number of convolutions.
        - series_id (int): the index of the series of convolutional layer in
        the network. For scoping purposes. Defaults to 0.
        - reuse (bool): whether we should reuse already instantiated variable
        or create new ones. Defaults to None.

    Returns:
        - tf tensor: the convolutional layer.

    """
    h = input_var
    for i in range(n_conv):
        name = "conv{series_id}_{layer_id}".format(
            series_id=series_id, layer_id=i + 1)
        h = constant_conv_layer(
            h, data_dict, name, reuse=reuse)
    h = max_pool_2x2(h)
    return h


def convolutional_layers(
    input_var, n_kernels, n_conv, series_id=0, reuse=None
):
    """Create a constant VGG like series of convolutional layers.

    This function creates a series of convolutional layers followed by a max
    pooling.

    Arguments:
        - input_var (tf tensor): the input of the convolutional layer.
        - data_dict (dict): the dictionnary containing the value for each
        variable.
        - n_kernels (int): the number of kernels in each convolution.
        - n_conv (int): the number of convolutions.
        - series_id (int): the index of the series of convolutional layer in
        the network. For scoping purposes. Defaults to 0.
        - reuse (bool): whether we should reuse already instantiated variable
        or create new ones. Defaults to None.

    Returns:
        - tf tensor: the convolutional layer.

    """
    h = input_var
    for i in range(n_conv):
        name = "conv{series_id}_{layer_id}".format(
            series_id=series_id, layer_id=i + 1)
        n_colors = int(h.shape[-1])
        h = conv_layer(h, n_colors, n_kernels, name, reuse=reuse)
    h = max_pool_2x2(h)
    return h


def vgg(
    x, data_dict, architecture_conv, architecture_fc, reuse=None
):
    """Create the vgg network."""
    h = x - VGG_MEAN
    for series_id, series in enumerate(architecture_conv):
        h = constant_convolutional_layers(
            h, data_dict, series["n_conv"],
            series_id=series_id + 1, reuse=reuse)
    for fc_id, fc in enumerate(architecture_fc):
        in_size = fc["in_size"]
        name = "fc{fc_id}".format(fc_id=fc_id + series_id + 2)
        h = constant_fc_layer(h, data_dict, in_size, name, reuse=reuse)
    last_name = "fc{fc_id}".format(fc_id=fc_id + series_id + 3)
    fc_dim = fc["fc_dim"]
    y = fc_layer(h, fc_dim, 2, last_name, reuse=reuse)
    return y


def zsc(
    x, architecture_conv, architecture_fc, reuse=None
):
    """
    Create the zsc network.

    ZSC stands for Zac simple CNN.
    """
    h = x - VGG_MEAN
    for series_id, series in enumerate(architecture_conv):
        h = convolutional_layers(
            h, series["n_kernels"], series["n_conv"],
            series_id=series_id + 1, reuse=reuse)
    for fc_id, fc in enumerate(architecture_fc):
        in_size = fc["in_size"]
        fc_dim = fc["fc_dim"]
        name = "fc{fc_id}".format(fc_id=fc_id + series_id + 2)
        h = fc_layer(h, in_size, fc_dim, name, reuse=reuse)
    last_name = "fc{fc_id}".format(fc_id=fc_id + series_id + 3)
    y = fc_layer(h, fc_dim, 2, last_name, reuse=reuse)
    return y
