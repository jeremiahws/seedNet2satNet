

'''seedNet2satNet/train_classifier.py

Build and train a SWCNN model on the SatNet dataset.
'''


import tensorflow as tf
import argparse
# from dataset.dataset_generator import DatasetGenerator
from dataset.hdf5_dataset_generator import DatasetGenerator
from dataset.swcnn_encoder import SWCNNEncoder
from dataset.tfrecords_parsing_functions import parse_swcnn_classification_data
from models.feature_extractor_zoo import VGG_like
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping
from tensorflow.keras.losses import sparse_categorical_crossentropy
import os


def train(FLAGS):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    tf.enable_eager_execution()
    input_shape = (FLAGS.window_size, FLAGS.window_size, 1)
    classification_model = VGG_like(input_shape, 2, softmax_output=True, dropout=True)
    encoder = SWCNNEncoder(FLAGS.window_size, FLAGS.stride, FLAGS.padding, FLAGS.bg2sat_ratio)
    optimizer = Adam(lr=FLAGS.learning_rate)

    with tf.device('/cpu:0'):
        # classification_train_generator = DatasetGenerator(FLAGS.classification_train_data,
        #                                                   num_windows=FLAGS.num_classifier_train_windows,
        #                                                   parse_function=parse_swcnn_classification_data,
        #                                                   augment=FLAGS.augment_training,
        #                                                   shuffle=FLAGS.shuffle_training,
        #                                                   batch_size=FLAGS.batch_size,
        #                                                   num_threads=FLAGS.num_dataset_threads,
        #                                                   buffer=FLAGS.dataset_buffer_size,
        #                                                   encoding_function=encoder.encode_for_swcnn_classifier,
        #                                                   cache_dataset_memory=FLAGS.cache_in_memory,
        #                                                   cache_dataset_file=FLAGS.cache_in_file,
        #                                                   cache_path="train_c_" + FLAGS.cache_name)
        #
        # classification_valid_generator = DatasetGenerator(FLAGS.classification_valid_data,
        #                                                   num_windows=FLAGS.num_classifier_valid_windows,
        #                                                   parse_function=parse_swcnn_classification_data,
        #                                                   augment=False,
        #                                                   shuffle=False,
        #                                                   batch_size=FLAGS.batch_size,
        #                                                   num_threads=FLAGS.num_dataset_threads,
        #                                                   buffer=FLAGS.dataset_buffer_size,
        #                                                   encoding_function=encoder.encode_for_swcnn_classifier,
        #                                                   cache_dataset_memory=FLAGS.cache_in_memory,
        #                                                   cache_dataset_file=FLAGS.cache_in_file,
        #                                                   cache_path="train_c_" + FLAGS.cache_name)

        classification_train_generator = DatasetGenerator(FLAGS.classification_train_data,
                                                          parse_function=parse_swcnn_classification_data,
                                                          augment=FLAGS.augment_training,
                                                          shuffle=FLAGS.shuffle_training,
                                                          batch_size=FLAGS.batch_size,
                                                          negpos_ratio=FLAGS.bg2sat_ratio)

        classification_valid_generator = DatasetGenerator(FLAGS.classification_valid_data,
                                                          parse_function=parse_swcnn_classification_data,
                                                          augment=False,
                                                          shuffle=False,
                                                          batch_size=FLAGS.batch_size,
                                                          negpos_ratio=FLAGS.bg2sat_ratio)

    classification_train_generator.generate()
    classification_valid_generator.generate()

    c_lr = ReduceLROnPlateau(patience=5)
    c_csv = CSVLogger('classifier_' + FLAGS.csv_file_name)
    c_ckpt = ModelCheckpoint('classifier_' + FLAGS.model_ckpt_name, save_best_only=True, save_weights_only=True)
    c_stop = EarlyStopping(patience=11)

    ce_loss = sparse_categorical_crossentropy

    classification_model.compile(optimizer=optimizer, loss=ce_loss, metrics=['acc'])

    # class_weights = {0: 1., 1: float(FLAGS.bg2sat_ratio)}
    class_weights = {0: 1., 1: 1.}

    train_generator = classification_train_generator.generate()
    valid_generator = classification_valid_generator.generate()

    # classification_model.fit(classification_train_generator,
    #                          steps_per_epoch=len(classification_train_generator),
    #                          epochs=FLAGS.num_training_epochs,
    #                          verbose=1,
    #                          callbacks=[c_lr, c_csv, c_ckpt, c_stop],
    #                          validation_data=classification_valid_generator,
    #                          validation_steps=len(classification_valid_generator),
    #                          class_weight=class_weights)
    classification_model.fit_generator(generator=train_generator,
                                       steps_per_epoch=len(classification_train_generator),
                                       epochs=FLAGS.num_training_epochs,
                                       verbose=1,
                                       callbacks=[c_lr, c_csv, c_ckpt, c_stop],
                                       validation_data=valid_generator,
                                       validation_steps=len(classification_valid_generator))#,
                                       # class_weight=class_weights)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--classification_train_data', type=str,
                        default=r'C:\Users\jsanders\Desktop\Github\seedNet2satNet\seedNet2satNet_windowsize_24_stride_3_padding_4_negposratio_1_train_classification.h5',
                        help='Path to data file containing the the training data for the classifier.')

    parser.add_argument('--classification_valid_data', type=str,
                        default=r'C:\Users\jsanders\Desktop\Github\seedNet2satNet\seedNet2satNet_windowsize_24_stride_3_padding_4_negposratio_1_valid_classification.h5',
                        help='Path to data file containing the the validation data for the classifier.')

    parser.add_argument('--csv_file_name', type=str,
                        default='train_script_test.csv',
                        help='Name of the CSV file to be logged during training.')

    parser.add_argument('--model_ckpt_name', type=str,
                        default='train_script_test.h5',
                        help='Name of the model checkpoint that should be stored during training.')

    parser.add_argument('--num_classifier_train_windows', type=int,
                        default=20938046,
                        help='Number of sub-windows in the train tfrecords file for the classifier.')

    parser.add_argument('--num_classifier_valid_windows', type=int,
                        default=2630327,
                        help='Number of sub-windows in the valid tfrecords file for the classifier.')

    parser.add_argument('--window_size', type=int,
                        default=24,
                        help='Size of sub-windows (in pixels).')

    parser.add_argument('--stride', type=int,
                        default=3,
                        help='Stride of the sliding window (in pixels).')

    parser.add_argument('--padding', type=int,
                        default=4,
                        help='Padding to apply to sub-windows to avoid edge cases (in pixels).')

    parser.add_argument('--bg2sat_ratio', type=int,
                        default=1,
                        help='Ratio of background:satellite sub-windows in the training dataset.')

    parser.add_argument('--cache_in_memory', type=bool,
                        default=True,
                        help='Should we cache the dataset in memory?')

    parser.add_argument('--cache_in_file', type=bool,
                        default=False,
                        help='Should we cache the dataset to file?')

    parser.add_argument('--cache_name', type=str,
                        default="n/a",
                        help='Name to use as part of the cache file.')

    parser.add_argument('--shuffle_training', type=bool,
                        default=True,
                        help='Should we shuffle the training set?')

    parser.add_argument('--augment_training', type=bool,
                        default=False,
                        help='Should we augment the training set?')

    parser.add_argument('--dataset_buffer_size', type=int,
                        default=256,
                        help='Number of images to prefetch in the input pipeline.')

    parser.add_argument('--batch_size', type=int,
                        default=5,
                        help='Batch size to use in training and validation.')

    parser.add_argument('--num_dataset_threads', type=int,
                        default=4,
                        help='Number of threads to be used by the input pipeline.')

    parser.add_argument('--gpu_list', type=str,
                        default="0",
                        help='GPUs to use with this model.')

    parser.add_argument('--learning_rate', type=float,
                        default=1e-4,
                        help='Initial learning rate.')

    parser.add_argument('--num_training_epochs', type=int,
                        default=5,
                        help='Number of epochs to train model.')

    # parse known arguements
    FLAGS, unparsed = parser.parse_known_args()

    train(FLAGS)
