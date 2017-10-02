import csv
import os.path as op

import numpy as np
import tensorflow as tf


def read_and_decode(filename_queue, shape=(224, 224, 3)):
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
    label = tf.one_hot(features["label"], 2)
    label = tf.cast(label, tf.float32)


    return feature, label


def inputs(
    tfrecord_files, batch_size, shape=(224, 224, 3), num_epochs=None,
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


def summary_writer(log_name, model_name, log_path, sess):
    """Create a summary writer."""
    if log_name is None:
        log_name = model_name
    log_path = op.join(log_path, log_name)
    writer = tf.summary.FileWriter(log_path, sess.graph)
    return writer


def metrics(ground_truth, predictions, suffix=""):
    """Create tf ops for positive metrics and add them to tf summary.

    The metrics are in order:
        - accuracy
        - precision
        - recall
        - F1 score
    """
    ones_like_actuals = tf.ones_like(ground_truth)
    zeros_like_actuals = tf.zeros_like(ground_truth)
    ones_like_predictions = tf.ones_like(predictions)
    zeros_like_predictions = tf.zeros_like(predictions)

    tp = tf.reduce_sum(
        tf.cast(
          tf.logical_and(
            tf.equal(ground_truth, ones_like_actuals),
            tf.equal(predictions, ones_like_predictions)
          ),
          "float"
        )
    )

    tn = tf.reduce_sum(
        tf.cast(
          tf.logical_and(
            tf.equal(ground_truth, zeros_like_actuals),
            tf.equal(predictions, zeros_like_predictions)
          ),
          "float"
        )
    )

    fp = tf.reduce_sum(
        tf.cast(
          tf.logical_and(
            tf.equal(ground_truth, zeros_like_actuals),
            tf.equal(predictions, ones_like_predictions)
          ),
          "float"
        )
    )

    fn = tf.reduce_sum(
        tf.cast(
          tf.logical_and(
            tf.equal(ground_truth, ones_like_actuals),
            tf.equal(predictions, zeros_like_predictions)
          ),
          "float"
        )
    )

    tpr = tp/(tp + fn)

    accuracy = (tp + tn)/(tp + fp + fn + tn)

    recall = tpr
    recall = tf.where(tf.is_nan(recall), tf.ones_like(recall), recall)
    precision = tp/(tp + fp)
    precision = tf.where(
        tf.is_nan(precision), tf.ones_like(precision), precision)
    f1_score = (2 * (precision * recall)) / (precision + recall)
    f1_score = tf.where(tf.is_nan(f1_score), tf.zeros_like(f1_score), f1_score)

    tf.summary.scalar("accuracy{}".format(suffix), accuracy)
    tf.summary.scalar("precision{}".format(suffix), precision)
    tf.summary.scalar("recall{}".format(suffix), recall)
    tf.summary.scalar("f1_score{}".format(suffix), f1_score)

    return accuracy, precision, recall, f1_score


def records(train_csv, test_csv):
    """List the train and test tfrecords."""
    with open(train_csv) as in_file:
        csv_reader = csv.reader(in_file)
        train_files = next(csv_reader)
    with open(test_csv) as in_file:
        csv_reader = csv.reader(in_file)
        test_files = next(csv_reader)
    return train_files, test_files
