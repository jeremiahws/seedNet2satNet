

'''seedNet2satNet/experiment_generator.py

Runs experiments across a defined set of hyperparameters. Essentially just a
script that calls the classifier and localizer train functions.
'''


import argparse
import os
import itertools


def classifier_main(FLAGS):
    classifier_train_window_numbers = {
        'windowsize_16_stride_3_padding_2_negpos_ratio_1': 1,
        'windowsize_16_stride_3_padding_2_negpos_ratio_5': 1,
        'windowsize_16_stride_3_padding_2_negpos_ratio_10': 1,
        'windowsize_16_stride_3_padding_4_negpos_ratio_1': 1,
        'windowsize_16_stride_3_padding_4_negpos_ratio_5': 1,
        'windowsize_16_stride_3_padding_4_negpos_ratio_10': 1,
        'windowsize_16_stride_3_padding_6_negpos_ratio_1': 1,
        'windowsize_16_stride_3_padding_6_negpos_ratio_5': 1,
        'windowsize_16_stride_3_padding_6_negpos_ratio_10': 1,
        'windowsize_20_stride_3_padding_2_negpos_ratio_1': 1,
        'windowsize_20_stride_3_padding_2_negpos_ratio_5': 1,
        'windowsize_20_stride_3_padding_2_negpos_ratio_10': 1,
        'windowsize_20_stride_3_padding_4_negpos_ratio_1': 1,
        'windowsize_20_stride_3_padding_4_negpos_ratio_5': 1,
        'windowsize_20_stride_3_padding_4_negpos_ratio_10': 1,
        'windowsize_20_stride_3_padding_6_negpos_ratio_1': 1,
        'windowsize_20_stride_3_padding_6_negpos_ratio_5': 1,
        'windowsize_20_stride_3_padding_6_negpos_ratio_10': 1,
        'windowsize_20_stride_3_padding_8_negpos_ratio_1': 1,
        'windowsize_20_stride_3_padding_8_negpos_ratio_5': 1,
        'windowsize_20_stride_3_padding_8_negpos_ratio_10': 1,
        'windowsize_24_stride_3_padding_2_negpos_ratio_1': 1,
        'windowsize_24_stride_3_padding_2_negpos_ratio_5': 1,
        'windowsize_24_stride_3_padding_2_negpos_ratio_10': 1,
        'windowsize_24_stride_3_padding_4_negpos_ratio_1': 1,
        'windowsize_24_stride_3_padding_4_negpos_ratio_5': 1,
        'windowsize_24_stride_3_padding_4_negpos_ratio_10': 1,
        'windowsize_24_stride_3_padding_6_negpos_ratio_1': 1,
        'windowsize_24_stride_3_padding_6_negpos_ratio_5': 1,
        'windowsize_24_stride_3_padding_6_negpos_ratio_10': 1,
        'windowsize_24_stride_3_padding_8_negpos_ratio_1': 1,
        'windowsize_24_stride_3_padding_8_negpos_ratio_5': 1,
        'windowsize_24_stride_3_padding_8_negpos_ratio_10': 1,
        'windowsize_24_stride_3_padding_10_negpos_ratio_1': 1,
        'windowsize_24_stride_3_padding_10_negpos_ratio_5': 1,
        'windowsize_24_stride_3_padding_10_negpos_ratio_10': 1,
        'windowsize_28_stride_3_padding_2_negpos_ratio_1': 1,
        'windowsize_28_stride_3_padding_2_negpos_ratio_5': 1,
        'windowsize_28_stride_3_padding_2_negpos_ratio_10': 1,
        'windowsize_28_stride_3_padding_4_negpos_ratio_1': 1,
        'windowsize_28_stride_3_padding_4_negpos_ratio_5': 1,
        'windowsize_28_stride_3_padding_4_negpos_ratio_10': 1,
        'windowsize_28_stride_3_padding_6_negpos_ratio_1': 1,
        'windowsize_28_stride_3_padding_6_negpos_ratio_5': 1,
        'windowsize_28_stride_3_padding_6_negpos_ratio_10': 1,
        'windowsize_28_stride_3_padding_8_negpos_ratio_1': 1,
        'windowsize_28_stride_3_padding_8_negpos_ratio_5': 1,
        'windowsize_28_stride_3_padding_8_negpos_ratio_10': 1,
        'windowsize_28_stride_3_padding_10_negpos_ratio_1': 1,
        'windowsize_28_stride_3_padding_10_negpos_ratio_5': 1,
        'windowsize_28_stride_3_padding_10_negpos_ratio_10': 1,
        'windowsize_28_stride_3_padding_12_negpos_ratio_1': 1,
        'windowsize_28_stride_3_padding_12_negpos_ratio_5': 1,
        'windowsize_28_stride_3_padding_12_negpos_ratio_10': 1,
        'windowsize_32_stride_3_padding_2_negpos_ratio_1': 1,
        'windowsize_32_stride_3_padding_2_negpos_ratio_5': 1,
        'windowsize_32_stride_3_padding_2_negpos_ratio_10': 1,
        'windowsize_32_stride_3_padding_4_negpos_ratio_1': 1,
        'windowsize_32_stride_3_padding_4_negpos_ratio_5': 1,
        'windowsize_32_stride_3_padding_4_negpos_ratio_10': 1,
        'windowsize_32_stride_3_padding_6_negpos_ratio_1': 1,
        'windowsize_32_stride_3_padding_6_negpos_ratio_5': 1,
        'windowsize_32_stride_3_padding_6_negpos_ratio_10': 1,
        'windowsize_32_stride_3_padding_8_negpos_ratio_1': 1,
        'windowsize_32_stride_3_padding_8_negpos_ratio_5': 1,
        'windowsize_32_stride_3_padding_8_negpos_ratio_10': 1,
        'windowsize_32_stride_3_padding_10_negpos_ratio_1': 1,
        'windowsize_32_stride_3_padding_10_negpos_ratio_5': 1,
        'windowsize_32_stride_3_padding_10_negpos_ratio_10': 1,
        'windowsize_32_stride_3_padding_12_negpos_ratio_1': 1,
        'windowsize_32_stride_3_padding_12_negpos_ratio_5': 1,
        'windowsize_32_stride_3_padding_12_negpos_ratio_10': 1,
        'windowsize_32_stride_3_padding_14_negpos_ratio_1': 241484,
        'windowsize_32_stride_3_padding_14_negpos_ratio_5': 756952,
        'windowsize_32_stride_3_padding_14_negpos_ratio_10': 1401287
    }

    classifier_valid_window_numbers = {
        'windowsize_16_stride_3_padding_2_negpos_ratio_1': 1,
        'windowsize_16_stride_3_padding_2_negpos_ratio_5': 1,
        'windowsize_16_stride_3_padding_2_negpos_ratio_10': 1,
        'windowsize_16_stride_3_padding_4_negpos_ratio_1': 1,
        'windowsize_16_stride_3_padding_4_negpos_ratio_5': 1,
        'windowsize_16_stride_3_padding_4_negpos_ratio_10': 1,
        'windowsize_16_stride_3_padding_6_negpos_ratio_1': 1,
        'windowsize_16_stride_3_padding_6_negpos_ratio_5': 1,
        'windowsize_16_stride_3_padding_6_negpos_ratio_10': 1,
        'windowsize_20_stride_3_padding_2_negpos_ratio_1': 1,
        'windowsize_20_stride_3_padding_2_negpos_ratio_5': 1,
        'windowsize_20_stride_3_padding_2_negpos_ratio_10': 1,
        'windowsize_20_stride_3_padding_4_negpos_ratio_1': 1,
        'windowsize_20_stride_3_padding_4_negpos_ratio_5': 1,
        'windowsize_20_stride_3_padding_4_negpos_ratio_10': 1,
        'windowsize_20_stride_3_padding_6_negpos_ratio_1': 1,
        'windowsize_20_stride_3_padding_6_negpos_ratio_5': 1,
        'windowsize_20_stride_3_padding_6_negpos_ratio_10': 1,
        'windowsize_20_stride_3_padding_8_negpos_ratio_1': 1,
        'windowsize_20_stride_3_padding_8_negpos_ratio_5': 1,
        'windowsize_20_stride_3_padding_8_negpos_ratio_10': 1,
        'windowsize_24_stride_3_padding_2_negpos_ratio_1': 1,
        'windowsize_24_stride_3_padding_2_negpos_ratio_5': 1,
        'windowsize_24_stride_3_padding_2_negpos_ratio_10': 1,
        'windowsize_24_stride_3_padding_4_negpos_ratio_1': 1,
        'windowsize_24_stride_3_padding_4_negpos_ratio_5': 1,
        'windowsize_24_stride_3_padding_4_negpos_ratio_10': 1,
        'windowsize_24_stride_3_padding_6_negpos_ratio_1': 1,
        'windowsize_24_stride_3_padding_6_negpos_ratio_5': 1,
        'windowsize_24_stride_3_padding_6_negpos_ratio_10': 1,
        'windowsize_24_stride_3_padding_8_negpos_ratio_1': 1,
        'windowsize_24_stride_3_padding_8_negpos_ratio_5': 1,
        'windowsize_24_stride_3_padding_8_negpos_ratio_10': 1,
        'windowsize_24_stride_3_padding_10_negpos_ratio_1': 1,
        'windowsize_24_stride_3_padding_10_negpos_ratio_5': 1,
        'windowsize_24_stride_3_padding_10_negpos_ratio_10': 1,
        'windowsize_28_stride_3_padding_2_negpos_ratio_1': 1,
        'windowsize_28_stride_3_padding_2_negpos_ratio_5': 1,
        'windowsize_28_stride_3_padding_2_negpos_ratio_10': 1,
        'windowsize_28_stride_3_padding_4_negpos_ratio_1': 1,
        'windowsize_28_stride_3_padding_4_negpos_ratio_5': 1,
        'windowsize_28_stride_3_padding_4_negpos_ratio_10': 1,
        'windowsize_28_stride_3_padding_6_negpos_ratio_1': 1,
        'windowsize_28_stride_3_padding_6_negpos_ratio_5': 1,
        'windowsize_28_stride_3_padding_6_negpos_ratio_10': 1,
        'windowsize_28_stride_3_padding_8_negpos_ratio_1': 1,
        'windowsize_28_stride_3_padding_8_negpos_ratio_5': 1,
        'windowsize_28_stride_3_padding_8_negpos_ratio_10': 1,
        'windowsize_28_stride_3_padding_10_negpos_ratio_1': 1,
        'windowsize_28_stride_3_padding_10_negpos_ratio_5': 1,
        'windowsize_28_stride_3_padding_10_negpos_ratio_10': 1,
        'windowsize_28_stride_3_padding_12_negpos_ratio_1': 1,
        'windowsize_28_stride_3_padding_12_negpos_ratio_5': 1,
        'windowsize_28_stride_3_padding_12_negpos_ratio_10': 1,
        'windowsize_32_stride_3_padding_2_negpos_ratio_1': 1,
        'windowsize_32_stride_3_padding_2_negpos_ratio_5': 1,
        'windowsize_32_stride_3_padding_2_negpos_ratio_10': 1,
        'windowsize_32_stride_3_padding_4_negpos_ratio_1': 1,
        'windowsize_32_stride_3_padding_4_negpos_ratio_5': 1,
        'windowsize_32_stride_3_padding_4_negpos_ratio_10': 1,
        'windowsize_32_stride_3_padding_6_negpos_ratio_1': 1,
        'windowsize_32_stride_3_padding_6_negpos_ratio_5': 1,
        'windowsize_32_stride_3_padding_6_negpos_ratio_10': 1,
        'windowsize_32_stride_3_padding_8_negpos_ratio_1': 1,
        'windowsize_32_stride_3_padding_8_negpos_ratio_5': 1,
        'windowsize_32_stride_3_padding_8_negpos_ratio_10': 1,
        'windowsize_32_stride_3_padding_10_negpos_ratio_1': 1,
        'windowsize_32_stride_3_padding_10_negpos_ratio_5': 1,
        'windowsize_32_stride_3_padding_10_negpos_ratio_10': 1,
        'windowsize_32_stride_3_padding_12_negpos_ratio_1': 1,
        'windowsize_32_stride_3_padding_12_negpos_ratio_5': 1,
        'windowsize_32_stride_3_padding_12_negpos_ratio_10': 1,
        'windowsize_32_stride_3_padding_14_negpos_ratio_1': 30234,
        'windowsize_32_stride_3_padding_14_negpos_ratio_5': 94634,
        'windowsize_32_stride_3_padding_14_negpos_ratio_10': 175134
    }

    paddings = FLAGS.padding.split(',')
    window_sizes = FLAGS.window_size.split(',')
    strides = FLAGS.stride.split(',')
    bg2sat_ratios = FLAGS.bg2sat_ratio.split(',')
    experiments = [window_sizes, strides, paddings, bg2sat_ratios]

    for experiment in itertools.product(*experiments):
        if int(experiment[0]) - 2 * int(experiment[2]) >= FLAGS.minimum_center\
                and int(experiment[1]) < int(experiment[0]) - 2 * int(experiment[2]):
            c_train_tfrecords = os.path.join(FLAGS.tfrecords_dir, 'seedNet2satNet_windowsize_{}_stride_{}_padding_{}_negposratio_{}_train_classification.tfrecords'.format(experiment[0], experiment[1], experiment[2], experiment[3]))
            c_valid_tfrecords = os.path.join(FLAGS.tfrecords_dir, 'seedNet2satNet_windowsize_{}_stride_{}_padding_{}_negposratio_{}_valid_classification.tfrecords'.format(experiment[0], experiment[1], experiment[2], experiment[3]))
            ckpt_file = 'seedNet2satNet_classifier_windowsize_{}_stride_{}_padding_{}_ratio_{}.h5'.format(experiment[0], experiment[1], experiment[2], experiment[3])
            csv_file = 'seedNet2satNet_classifier_windowsize_{}_stride_{}_padding_{}_ratio_{}.csv'.format(experiment[0], experiment[1], experiment[2], experiment[3])

            dict_call = 'windowsize_{}_stride_{}_padding_{}_negpos_ratio_{}'.format(experiment[0], experiment[1], experiment[2], experiment[3])
            create_command = 'python train_classifier.py'\
                             + ' --classification_train_tfrecords={}'.format(c_train_tfrecords)\
                             + ' --classification_valid_tfrecords={}'.format(c_valid_tfrecords)\
                             + ' --csv_file_name={}'.format(csv_file)\
                             + ' --model_ckpt_name={}'.format(ckpt_file)\
                             + ' --num_classifier_train_windows={}'.format(classifier_train_window_numbers[dict_call])\
                             + ' --num_classifier_valid_windows={}'.format(classifier_valid_window_numbers[dict_call])\
                             + ' --window_size={}'.format(experiment[0])\
                             + ' --stride={}'.format(experiment[1])\
                             + ' --padding={}'.format(experiment[2])\
                             + ' --bg2sat_ratio={}'.format(experiment[3])\
                             + ' --shuffle_training={}'.format(FLAGS.shuffle_training)\
                             + ' --batch_size={}'.format(FLAGS.batch_size)\
                             + ' --learning_rate={}'.format(FLAGS.learning_rate)\
                             + ' --num_training_epochs={}'.format(FLAGS.training_epochs)

            os.system(create_command)


def localizer_main(FLAGS):
    localizer_train_window_numbers = {
        'windowsize_16_stride_3_padding_2': 1,
        'windowsize_16_stride_3_padding_4': 1,
        'windowsize_16_stride_3_padding_6': 1,
        'windowsize_20_stride_3_padding_2': 1,
        'windowsize_20_stride_3_padding_4': 1,
        'windowsize_20_stride_3_padding_6': 1,
        'windowsize_20_stride_3_padding_8': 1,
        'windowsize_24_stride_3_padding_2': 1,
        'windowsize_24_stride_3_padding_4': 1,
        'windowsize_24_stride_3_padding_6': 1,
        'windowsize_24_stride_3_padding_8': 1,
        'windowsize_24_stride_3_padding_10': 1,
        'windowsize_28_stride_3_padding_2': 1,
        'windowsize_28_stride_3_padding_4': 1,
        'windowsize_28_stride_3_padding_6': 1,
        'windowsize_28_stride_3_padding_8': 1,
        'windowsize_28_stride_3_padding_10': 1,
        'windowsize_28_stride_3_padding_12': 1,
        'windowsize_32_stride_3_padding_2': 1,
        'windowsize_32_stride_3_padding_4': 1,
        'windowsize_32_stride_3_padding_6': 1,
        'windowsize_32_stride_3_padding_8': 1,
        'windowsize_32_stride_3_padding_10': 1,
        'windowsize_32_stride_3_padding_12': 1,
        'windowsize_32_stride_3_padding_14': 112617
    }

    localizer_valid_window_numbers = {
        'windowsize_16_stride_3_padding_2': 1,
        'windowsize_16_stride_3_padding_4': 1,
        'windowsize_16_stride_3_padding_6': 1,
        'windowsize_20_stride_3_padding_2': 1,
        'windowsize_20_stride_3_padding_4': 1,
        'windowsize_20_stride_3_padding_6': 1,
        'windowsize_20_stride_3_padding_8': 1,
        'windowsize_24_stride_3_padding_2': 1,
        'windowsize_24_stride_3_padding_4': 1,
        'windowsize_24_stride_3_padding_6': 1,
        'windowsize_24_stride_3_padding_8': 1,
        'windowsize_24_stride_3_padding_10': 1,
        'windowsize_28_stride_3_padding_2': 1,
        'windowsize_28_stride_3_padding_4': 1,
        'windowsize_28_stride_3_padding_6': 1,
        'windowsize_28_stride_3_padding_8': 1,
        'windowsize_28_stride_3_padding_10': 1,
        'windowsize_28_stride_3_padding_12': 1,
        'windowsize_32_stride_3_padding_2': 1,
        'windowsize_32_stride_3_padding_4': 1,
        'windowsize_32_stride_3_padding_6': 1,
        'windowsize_32_stride_3_padding_8': 1,
        'windowsize_32_stride_3_padding_10': 1,
        'windowsize_32_stride_3_padding_12': 1,
        'windowsize_32_stride_3_padding_14': 14134
    }

    paddings = FLAGS.padding.split(',')
    window_sizes = FLAGS.window_size.split(',')
    strides = FLAGS.stride.split(',')
    experiments = [window_sizes, strides, paddings]

    for experiment in itertools.product(*experiments):
        if int(experiment[0]) - 2 * int(experiment[2]) >= FLAGS.minimum_center\
                and int(experiment[1]) < int(experiment[0]) - 2 * int(experiment[2]):
            l_train_tfrecords = os.path.join(FLAGS.tfrecords_dir, 'seedNet2satNet_windowsize_{}_stride_{}_padding_{}_train_localization.tfrecords'.format(experiment[0], experiment[1], experiment[2]))
            l_valid_tfrecords = os.path.join(FLAGS.tfrecords_dir, 'seedNet2satNet_windowsize_{}_stride_{}_padding_{}_valid_localization.tfrecords'.format(experiment[0], experiment[1], experiment[2]))
            ckpt_file = 'seedNet2satNet_localizer_windowsize_{}_stride_{}_padding_{}.h5'.format(experiment[0], experiment[1], experiment[2])
            csv_file = 'seedNet2satNet_localizer_windowsize_{}_stride_{}_padding_{}.csv'.format(experiment[0], experiment[1], experiment[2])

            dict_call = 'windowsize_{}_stride_{}_padding_{}'.format(experiment[0], experiment[1], experiment[2])
            create_command = 'python train_localizer.py'\
                             + ' --localization_train_tfrecords={}'.format(l_train_tfrecords)\
                             + ' --localization_valid_tfrecords={}'.format(l_valid_tfrecords)\
                             + ' --csv_file_name={}'.format(csv_file)\
                             + ' --model_ckpt_name={}'.format(ckpt_file)\
                             + ' --num_localizer_train_windows={}'.format(localizer_train_window_numbers[dict_call])\
                             + ' --num_localizer_valid_windows={}'.format(localizer_valid_window_numbers[dict_call])\
                             + ' --window_size={}'.format(experiment[0])\
                             + ' --stride={}'.format(experiment[1])\
                             + ' --padding={}'.format(experiment[2])\
                             + ' --shuffle_training={}'.format(FLAGS.shuffle_training)\
                             + ' --batch_size={}'.format(FLAGS.batch_size)\
                             + ' --learning_rate={}'.format(FLAGS.learning_rate)\
                             + ' --num_training_epochs={}'.format(FLAGS.training_epochs)

            os.system(create_command)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--tfrecords_dir', type=str,
                        default=r'C:\Users\jsanders\Desktop\data\seednet2satnet\tfrecords',
                        help='Path to the directory containing all of the tfrecords files for the subwindows.')

    parser.add_argument('--window_size', type=str,
                        default='32',
                        help='Size of sub-windows (in pixels); single value or multiple values separated by a comma.')

    parser.add_argument('--stride', type=str,
                        default='3',
                        help='Stride of the sliding window (in pixels); single value or multiple values separated by a comma.')

    parser.add_argument('--padding', type=str,
                        default='14',
                        help='Padding to apply to sub-windows to avoid edge cases (in pixels); single value or multiple values separated by a comma.')

    parser.add_argument('--bg2sat_ratio', type=str,
                        default='1,5,10',
                        help='Ratio of background:satellite sub-windows in the training dataset.')

    parser.add_argument('--minimum_center', type=int,
                        default=4,
                        help='Minimum of the central coverage of the sub-windows.')

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
                        default=512,
                        help='Batch size to use in training and validation.')

    parser.add_argument('--num_dataset_threads', type=int,
                        default=4,
                        help='Number of threads to be used by the input pipeline.')

    parser.add_argument('--gpu_list', type=str,
                        default="1",
                        help='GPUs to use with this model.')

    parser.add_argument('--learning_rate', type=float,
                        default=1e-4,
                        help='Initial learning rate.')

    parser.add_argument('--training_epochs', type=int,
                        default=1000,
                        help='Number of epochs to train model.')

    # parse known arguements
    FLAGS, unparsed = parser.parse_known_args()

    classifier_main(FLAGS)
    localizer_main(FLAGS)
