import numpy as np
import tensorflow as tf


def read_and_decode(filename_queue, shape=(244, 244, 3)):
    """Read and decode a specific queue.

    This function takes as input a queue and deals with one example from it. It
    then returns the well-shaped example. Inspired from:
    https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/examples/how_tos/reading_data/fully_connected_reader.py#L47
    """
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          "image_raw": tf.FixedLenFeature([], tf.string),
          "label": tf.FixedLenFeature([], tf.int64),
      })

    feature = tf.decode_raw(features["image_raw"], tf.float64)
    feature.set_shape([np.prod(shape)])
    feature = tf.reshape(feature, shape)
    feature = tf.where(tf.is_nan(feature), tf.zeros_like(feature), feature)

    # This is where some normalization can be done. For now let's try without
    # it.
    feature = tf.cast(feature, tf.float32)

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features["label"], tf.float32)
    # label = tf.one_hot(label, 2)  # since we only have two classes, one label
    # is enough.

    return feature, label


def inputs(
    tfrecord_files, batch_size, shape=(244, 244, 3), num_epochs=None,
    n_threads=2
):
    """Read input data num_epochs times.

    Arguments:
    - tfrecord_files (list): The paths to the tfrecord_files where the data is.
    - batch_size (int): Number of examples per returned batch.
    - shape (tuple): the shape of the inputs. Defaults to (244, 244, 3).
    - num_epochs (int): Number of times to read the input data, or 0/None to
    train forever. Defaults to None.
    - n_threads (int): the number of threads in which to run the queue.
    Defaults to 2.

    Returns:
    - features is a float tensor with shape [batch_size, *shape].
    - labels is an int32 tensor with shape [batch_size, 2].
    Note that a tf.train.QueueRunner is added to the graph, which
    must be run using e.g. tf.train.start_queue_runners().

    """
    with tf.name_scope("input"):
        filename_queue = tf.train.string_input_producer(
            tfrecord_files, num_epochs=num_epochs)

        feature, label = read_and_decode(filename_queue, shape=shape)

        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.
        features, sparse_labels = tf.train.shuffle_batch(
            [feature, label], batch_size=batch_size, num_threads=n_threads,
            capacity=1000 + (n_threads + 1) * batch_size,
            # Ensures a minimum amount of shuffling of examples.
            min_after_dequeue=1000)

    return features, sparse_labels
