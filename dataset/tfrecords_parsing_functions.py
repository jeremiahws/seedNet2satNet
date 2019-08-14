

'''seedNet2satNet/dataset/tfrecords_parsing_function.py

Parsing functions for the dataset generator.
'''


import tensorflow as tf


def parse_swcnn_classification_data(example_proto):
    """
    This is the first step of the generator/augmentation chain. Reading the raw file out of the TFRecord is fairly
    straight-forward.
    :param example_proto: Example from a TFRecord file
    :return: The raw image and padded bounding boxes corresponding to this TFRecord example.
    """
    # Define how to parse the example
    features = {
        "filename": tf.VarLenFeature(dtype=tf.string),
        "window_height": tf.FixedLenFeature([], dtype=tf.int64),
        "window_width": tf.FixedLenFeature([], dtype=tf.int64),
        "window": tf.VarLenFeature(dtype=tf.string),
        "annotation": tf.VarLenFeature(dtype=tf.int64)
    }

    # Parse the example
    features_parsed = tf.parse_single_example(serialized=example_proto, features=features)
    height = tf.cast(features_parsed['window_height'], tf.int32)
    width = tf.cast(features_parsed['window_width'], tf.int32)

    filename = tf.cast(tf.sparse_tensor_to_dense(features_parsed['filename'], default_value=""), tf.string)
    annotations = tf.cast(tf.sparse_tensor_to_dense(features_parsed['annotation']), tf.int32)

    # get the sub-windows
    windows = tf.sparse_tensor_to_dense(features_parsed['window'], default_value="")
    windows = tf.decode_raw(windows, tf.uint16)
    windows = tf.reshape(windows, [height, width, 1])
    #windows = tf.image.per_image_standardization(windows)
    windows = tf.divide(tf.cast(windows, tf.float32), tf.constant(65535.0))

    # always return filename (let the user ignore it downstream)
    return windows, annotations, filename


def parse_swcnn_localization_data(example_proto):
    """
    This is the first step of the generator/augmentation chain. Reading the raw file out of the TFRecord is fairly
    straight-forward.
    :param example_proto: Example from a TFRecord file
    :return: The raw image and padded bounding boxes corresponding to this TFRecord example.
    """
    # Define how to parse the example
    features = {
        "filename": tf.VarLenFeature(tf.string),
        "window_height": tf.FixedLenFeature([], dtype=tf.int64),
        "window_width": tf.FixedLenFeature([], dtype=tf.int64),
        "window": tf.VarLenFeature(dtype=tf.string),
        "y_c": tf.VarLenFeature(dtype=tf.float32),
        "x_c": tf.VarLenFeature(dtype=tf.float32)
    }

    # Parse the example
    features_parsed = tf.parse_single_example(serialized=example_proto, features=features)
    height = tf.cast(features_parsed['window_height'], tf.int32)
    width = tf.cast(features_parsed['window_width'], tf.int32)

    filename = tf.cast(tf.sparse_tensor_to_dense(features_parsed['filename'], default_value=""), tf.string)
    y_c = tf.cast(tf.sparse_tensor_to_dense(features_parsed['y_c']), tf.float32)
    x_c = tf.cast(tf.sparse_tensor_to_dense(features_parsed['x_c']), tf.float32)
    annotations = tf.concat([y_c, x_c], 0)

    # get the sub-windows
    windows = tf.sparse_tensor_to_dense(features_parsed['window'], default_value="")
    windows = tf.decode_raw(windows, tf.uint16)
    windows = tf.reshape(windows, [height, width, 1])
    #windows = tf.image.per_image_standardization(windows)
    windows = tf.divide(tf.cast(windows, tf.float32), tf.constant(65535.0))

    # always return filename (let the user ignore it downstream)
    return windows, annotations, filename
