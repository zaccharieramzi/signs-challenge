from os.path import join
import time

import numpy as np
import tensorflow as tf

from nn import inputs, summary_writer, metrics, records
from model import vgg, zsc


# Datasets aggregation
train_csv = "data/train.csv"
test_csv = "data/test.csv"
train_tfrecord_files, test_tfrecord_files = records(train_csv, test_csv)


# Parameters
# Architecture
# VGG-16
architecture_conv = [
    {"n_conv": 2, "n_kernels": 64},
    {"n_conv": 2, "n_kernels": 128},
    {"n_conv": 3, "n_kernels": 256},
    {"n_conv": 3, "n_kernels": 512},
    {"n_conv": 3, "n_kernels": 512}
]
architecture_fc = [
    {"in_size": 25088, "fc_dim": 4096},
    {"in_size": 4096, "fc_dim": 4096}
]


# Parameters
# Architecture
# ZSC
architecture_conv = [
    {"n_conv": 1, "n_kernels": 16},
    {"n_conv": 1, "n_kernels": 32},
    {"n_conv": 1, "n_kernels": 64},
    {"n_conv": 1, "n_kernels": 128},
    {"n_conv": 1, "n_kernels": 128}
]
architecture_fc = [
    {"in_size": 6272, "fc_dim": 512},
    {"in_size": 512, "fc_dim": 512}
]


# Parameters
# Training
batch_size = 128
n_epochs = 4
n_threads = 2
l_rate = 0.01
keep_prob = 0.5
# Values
data_dict_path = "models/vgg16.npy"
# data_dict = np.load(data_dict_path, encoding="latin1").item()
# Saving and summary
save_path = "models"
model_name = "vgg-fine-tuned"
log_path = "logs"
log_name = None
summary_rate = 35
saving_rate = 10


tf.reset_default_graph()


with tf.name_scope("train"):
    x, y_ = inputs(
        train_tfrecord_files, batch_size,
        num_epochs=n_epochs, n_threads=n_threads)
    y_ = tf.reshape(y_, [-1, 2])

    y = zsc(x, architecture_conv, architecture_fc, keep_prob=keep_prob)

with tf.name_scope("test"):
    x_test, y_test_ = inputs(
        test_tfrecord_files, batch_size,
        num_epochs=None, n_threads=n_threads)
    y_test_ = tf.reshape(y_test_, [-1, 2])

    y_test = zsc(x_test, architecture_conv, architecture_fc, reuse=True)


# When training vgg-16, we fine tune the last layer.
# train_var_names = ["fc8/fc8_weights:0", "fc8/fc8_biases:0"]
# train_vars = [var for var in tf.trainable_variables() if var.name in train_var_names]


with tf.name_scope("loss"):
    # Define loss function
    cross_entropy = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(
            labels=y_, logits=y))
    cross_entropy_test = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(
            labels=y_test_, logits=y_test))
    # Define Optimizer
    train_step = tf.train.AdamOptimizer(l_rate).minimize(cross_entropy)
    with tf.name_scope("loss_summary"):
        tf.summary.scalar("loss", cross_entropy)
        tf.summary.scalar("loss_test", cross_entropy_test)


# Metrics
with tf.name_scope("metrics"):
    # Train
    ground_truth = tf.argmax(y_, axis=1)
    predictions = tf.argmax(y, axis=1)
    accuracy, _, _, _ = metrics(ground_truth, predictions)
    # Test
    ground_truth_test = tf.argmax(y_test_, axis=1)
    predictions_test = tf.argmax(y_test, axis=1)
    metrics(ground_truth_test, predictions_test, "_test")


# Start session
sess = tf.InteractiveSession()
init_op = tf.group(
    tf.global_variables_initializer(),
    tf.local_variables_initializer())
sess.run(init_op)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)


# Saving
saver = tf.train.Saver()
model_name = "vgg_fine_tuned"
saving_path = join(save_path, model_name)


# Summaries
merged = tf.summary.merge_all()
writer = summary_writer(log_name, model_name, log_path, sess)


try:
    step = 0
    while not coord.should_stop():
        print(step)
        step_start_time = time.time()
        # main part of this loop here: train the CNN and get its accuracy.
        sess.run([train_step])
        step_duration = time.time() - step_start_time
        # Print an overview fairly often.
        if step % summary_rate == 0:
            kwargs = dict()
            kwargs["options"] = tf.RunOptions(
                trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            kwargs["run_metadata"] = run_metadata
            accuracy_value, summary = sess.run(
                [accuracy, merged], **kwargs)
            writer.add_summary(summary, step)
            writer.add_run_metadata(run_metadata, "step {}".format(step))
            print("Step {step}: accuracy = {acc} ({duration} sec)".format(
                step=step,
                acc=accuracy_value,
                duration=step_duration)
            )
        if step % saving_rate == 0:
            saver.save(sess, saving_path, global_step=step)
        step += 1
except tf.errors.OutOfRangeError:
    print("Done training for {n_epochs} epochs, {n_steps} steps.".format(
        n_epochs=n_epochs,
        n_steps=step
    ))
finally:
    # When done, ask the threads to stop.
    coord.request_stop()


# Wait for threads to finish.
coord.join(threads)
sess.close()
