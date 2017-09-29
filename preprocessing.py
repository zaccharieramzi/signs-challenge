import numpy as np
import tensorflow as tf


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def pad_image(image, shape=(244, 244, 3)):
    padded_image = np.zeros(shape)
    first_dim = image.shape[0]
    first_dim = first_dim - (first_dim % 2)
    second_dim = image.shape[1]
    second_dim = second_dim - (second_dim % 2)
    if first_dim < shape[0]:
        padded_image[
            (shape[0] - first_dim)//2:shape[0]-(shape[0] - first_dim)//2,
            (shape[1] - second_dim)//2:shape[1]-(shape[1] - second_dim)//2,
            :
        ] = image[:first_dim, :second_dim, :]
    else:
        # for now we crop the image
        # TODO find way to resize image properly
        padded_image = image[
            (first_dim - shape[0])//2:first_dim-(first_dim - shape[0])//2,
            (second_dim - shape[1])//2:second_dim-(second_dim - shape[1])//2,
            :
        ]
    return padded_image
