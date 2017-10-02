import numpy as np
import tensorflow as tf


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def pad_image(image, shape=(224, 224, 3)):
    padded_image = np.zeros(shape)
    first_dim = image.shape[0]
    first_dim = first_dim - (first_dim % 2)
    if first_dim > shape[0]:
        fit_image = image[
            (first_dim - shape[0])//2:first_dim-(first_dim - shape[0])//2, :, :
        ]
        first_dim = shape[0]
    else:
        fit_image = image[:first_dim, :, :]
    second_dim = image.shape[1]
    second_dim = second_dim - (second_dim % 2)
    if second_dim > shape[1]:
        fit_2_image = fit_image[
            :,
            (second_dim - shape[1])//2:second_dim-(second_dim - shape[1])//2,
            :
        ]
        second_dim = shape[1]
    else:
        fit_2_image = fit_image[:, :second_dim, :]
    if first_dim <= shape[0] or second_dim <= shape[1]:
        padded_image[
            (shape[0] - first_dim)//2:shape[0]-(shape[0] - first_dim)//2,
            (shape[1] - second_dim)//2:shape[1]-(shape[1] - second_dim)//2,
            :
        ] = fit_2_image
    else:
        padded_image = fit_2_image
    return padded_image
