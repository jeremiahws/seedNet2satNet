

'''seedNet2satNet/dataset/swcnn_encoder.py

Encodes sub-window data for the dataset generator that feeds a SWCNN model.
'''


import tensorflow as tf


class SWCNNEncoder(object):
    def __init__(self,
                 window_size,
                 stride,
                 padding,
                 neg_pos_ratio):
        '''Encoder for a sliding-window CNN.

        Most of the encoding has already been done in the sub-window extraction. This can be changed in the future
        to perform the encoding here. For now, we'll just pass the pre-encoded information through. Also, sliding-window
        CNNs inherently don't require much encoding of the windows or targets.

        :param window_size: size of the sub-windows
        :param stride: stride of the sliding-window
        :param padding: padding to apply to the windows to avoid edge cases
        :param neg_pos_ratio: ratio of background to object windows. Should be >= 1
        '''
        self.window_size = window_size
        self.stride = stride
        self.padding = padding
        self.neg_pos_ratio = neg_pos_ratio

    def encode_for_swcnn_classifier(self, windows, annotations, filename=None):
        '''Encode for the first stage of the model - classification.

        :param windows: the sub-windows
        :param annotations: annotations of the sub-window classes
        :param filename: filename form the image where the sub-window originated from
        :return: encoded windows and annotations
        '''

        return tf.cast(windows, tf.float32), annotations

    def encode_for_swcnn_localizer(self, windows, annotations, filename=None):
        '''Encode for the second stage of the model - classification.

        :param windows: the sub-windows
        :param annotations: annotations of the object locations within the sub-window
        :param filename: filename form the image where the sub-window originated from
        :return: encoded windows and annotations
        '''

        return tf.cast(windows, tf.float32), annotations
