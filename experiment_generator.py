

'''seedNet2satNet/experiment_generator.py

Runs experiments across a defined set of hyperparameters. Essentially just a
script that calls the classifier and localizer train functions.
'''


import argparse
import os
import itertools


def classifier_main(FLAGS):
    classifier_train_window_numbers = {
        'windowsize_16_stride_3_padding_2_negpos_ratio_1': 2124922,
        'windowsize_16_stride_3_padding_2_negpos_ratio_5': 6406358,
        'windowsize_16_stride_3_padding_2_negpos_ratio_10': 11758153,
        'windowsize_16_stride_3_padding_4_negpos_ratio_1': 952544,
        'windowsize_16_stride_3_padding_4_negpos_ratio_5': 2889328,
        'windowsize_16_stride_3_padding_4_negpos_ratio_10': 5310308,
        'windowsize_16_stride_3_padding_6_negpos_ratio_1': 239239,
        'windowsize_16_stride_3_padding_6_negpos_ratio_5': 749503,
        'windowsize_16_stride_3_padding_6_negpos_ratio_10': 1387333,
        'windowsize_20_stride_3_padding_2_negpos_ratio_1': 3772330,
        'windowsize_20_stride_3_padding_2_negpos_ratio_5': 11348582,
        'windowsize_20_stride_3_padding_2_negpos_ratio_10': 20818897,
        'windowsize_20_stride_3_padding_4_negpos_ratio_1': 2126300,
        'windowsize_20_stride_3_padding_4_negpos_ratio_5': 6410596,
        'windowsize_20_stride_3_padding_4_negpos_ratio_10': 11765966,
        'windowsize_20_stride_3_padding_6_negpos_ratio_1': 946307,
        'windowsize_20_stride_3_padding_6_negpos_ratio_5': 2870707,
        'windowsize_20_stride_3_padding_6_negpos_ratio_10': 5276207,
        'windowsize_20_stride_3_padding_8_negpos_ratio_1': 242112,
        'windowsize_20_stride_3_padding_8_negpos_ratio_5': 758236,
        'windowsize_20_stride_3_padding_8_negpos_ratio_10': 1403391,
        'windowsize_24_stride_3_padding_2_negpos_ratio_1': 5845846,
        'windowsize_24_stride_3_padding_2_negpos_ratio_5': 17569130,
        'windowsize_24_stride_3_padding_2_negpos_ratio_10': 32223235,
        'windowsize_24_stride_3_padding_4_negpos_ratio_1': 3787060,
        'windowsize_24_stride_3_padding_4_negpos_ratio_5': 11392876,
        'windowsize_24_stride_3_padding_4_negpos_ratio_10': 20900146,
        'windowsize_24_stride_3_padding_6_negpos_ratio_1': 2124577,
        'windowsize_24_stride_3_padding_6_negpos_ratio_5': 6405517,
        'windowsize_24_stride_3_padding_6_negpos_ratio_10': 11756692,
        'windowsize_24_stride_3_padding_8_negpos_ratio_1': 939186,
        'windowsize_24_stride_3_padding_8_negpos_ratio_5': 2849458,
        'windowsize_24_stride_3_padding_8_negpos_ratio_10': 5237298,
        'windowsize_24_stride_3_padding_10_negpos_ratio_1': 245260,
        'windowsize_24_stride_3_padding_10_negpos_ratio_5': 767840,
        'windowsize_24_stride_3_padding_10_negpos_ratio_10': 1421065,
        'windowsize_28_stride_3_padding_2_negpos_ratio_1': 8424460,
        'windowsize_28_stride_3_padding_2_negpos_ratio_5': 25304972,
        'windowsize_28_stride_3_padding_2_negpos_ratio_10': 46405612,
        'windowsize_28_stride_3_padding_4_negpos_ratio_1': 5881424,
        'windowsize_28_stride_3_padding_4_negpos_ratio_5': 17675968,
        'windowsize_28_stride_3_padding_4_negpos_ratio_10': 32419148,
        'windowsize_28_stride_3_padding_6_negpos_ratio_1': 3761859,
        'windowsize_28_stride_3_padding_6_negpos_ratio_5': 11317363,
        'windowsize_28_stride_3_padding_6_negpos_ratio_10': 20761743,
        'windowsize_28_stride_3_padding_8_negpos_ratio_1': 2123078,
        'windowsize_28_stride_3_padding_8_negpos_ratio_5': 6401134,
        'windowsize_28_stride_3_padding_8_negpos_ratio_10': 11748704,
        'windowsize_28_stride_3_padding_10_negpos_ratio_1': 949680,
        'windowsize_28_stride_3_padding_10_negpos_ratio_5': 2881100,
        'windowsize_28_stride_3_padding_10_negpos_ratio_10': 5295375,
        'windowsize_28_stride_3_padding_12_negpos_ratio_1': 238666,
        'windowsize_28_stride_3_padding_12_negpos_ratio_5': 748254,
        'windowsize_28_stride_3_padding_12_negpos_ratio_10': 1385239,
        'windowsize_32_stride_3_padding_2_negpos_ratio_1': 11444656,
        'windowsize_32_stride_3_padding_2_negpos_ratio_5': 34365560,
        'windowsize_32_stride_3_padding_2_negpos_ratio_10': 63016690,
        'windowsize_32_stride_3_padding_4_negpos_ratio_1': 8432846,
        'windowsize_32_stride_3_padding_4_negpos_ratio_5': 25330234,
        'windowsize_32_stride_3_padding_4_negpos_ratio_10': 46451969,
        'windowsize_32_stride_3_padding_6_negpos_ratio_1': 5870621,
        'windowsize_32_stride_3_padding_6_negpos_ratio_5': 17643649,
        'windowsize_32_stride_3_padding_6_negpos_ratio_10': 32359934,
        'windowsize_32_stride_3_padding_8_negpos_ratio_1': 3770878,
        'windowsize_32_stride_3_padding_8_negpos_ratio_5': 11344534,
        'windowsize_32_stride_3_padding_8_negpos_ratio_10': 20811604,
        'windowsize_32_stride_3_padding_10_negpos_ratio_1': 2120176,
        'windowsize_32_stride_3_padding_10_negpos_ratio_5': 6392588,
        'windowsize_32_stride_3_padding_10_negpos_ratio_10': 11733103,
        'windowsize_32_stride_3_padding_12_negpos_ratio_1': 942538,
        'windowsize_32_stride_3_padding_12_negpos_ratio_5': 2859870,
        'windowsize_32_stride_3_padding_12_negpos_ratio_10': 5256535,
        'windowsize_32_stride_3_padding_14_negpos_ratio_1': 241484,
        'windowsize_32_stride_3_padding_14_negpos_ratio_5': 756952,
        'windowsize_32_stride_3_padding_14_negpos_ratio_10': 1401287
    }

    classifier_valid_window_numbers = {
        'windowsize_16_stride_3_padding_2_negpos_ratio_1': 268071,
        'windowsize_16_stride_3_padding_2_negpos_ratio_5': 808047,
        'windowsize_16_stride_3_padding_2_negpos_ratio_10': 1483017,
        'windowsize_16_stride_3_padding_4_negpos_ratio_1': 119047,
        'windowsize_16_stride_3_padding_4_negpos_ratio_5': 360983,
        'windowsize_16_stride_3_padding_4_negpos_ratio_10': 663403,
        'windowsize_16_stride_3_padding_6_negpos_ratio_1': 30467,
        'windowsize_16_stride_3_padding_6_negpos_ratio_5': 95263,
        'windowsize_16_stride_3_padding_6_negpos_ratio_10': 176258,
        'windowsize_20_stride_3_padding_2_negpos_ratio_1': 474923,
        'windowsize_20_stride_3_padding_2_negpos_ratio_5': 1428603,
        'windowsize_20_stride_3_padding_2_negpos_ratio_10': 2620703,
        'windowsize_20_stride_3_padding_4_negpos_ratio_1': 268231,
        'windowsize_20_stride_3_padding_4_negpos_ratio_5': 808535,
        'windowsize_20_stride_3_padding_4_negpos_ratio_10': 1483915,
        'windowsize_20_stride_3_padding_6_negpos_ratio_1': 119439,
        'windowsize_20_stride_3_padding_6_negpos_ratio_5': 362179,
        'windowsize_20_stride_3_padding_6_negpos_ratio_10': 665604,
        'windowsize_20_stride_3_padding_8_negpos_ratio_1': 30322,
        'windowsize_20_stride_3_padding_8_negpos_ratio_5': 94842,
        'windowsize_20_stride_3_padding_8_negpos_ratio_10': 175492,
        'windowsize_24_stride_3_padding_2_negpos_ratio_1': 738055,
        'windowsize_24_stride_3_padding_2_negpos_ratio_5': 2217999,
        'windowsize_24_stride_3_padding_2_negpos_ratio_10': 4067929,
        'windowsize_24_stride_3_padding_4_negpos_ratio_1': 475997,
        'windowsize_24_stride_3_padding_4_negpos_ratio_5': 1431833,
        'windowsize_24_stride_3_padding_4_negpos_ratio_10': 2626628,
        'windowsize_24_stride_3_padding_6_negpos_ratio_1': 267741,
        'windowsize_24_stride_3_padding_6_negpos_ratio_5': 807085,
        'windowsize_24_stride_3_padding_6_negpos_ratio_10': 1481265,
        'windowsize_24_stride_3_padding_8_negpos_ratio_1': 119014,
        'windowsize_24_stride_3_padding_8_negpos_ratio_5': 360918,
        'windowsize_24_stride_3_padding_8_negpos_ratio_10': 663298,
        'windowsize_24_stride_3_padding_10_negpos_ratio_1': 30426,
        'windowsize_24_stride_3_padding_10_negpos_ratio_5': 95158,
        'windowsize_24_stride_3_padding_10_negpos_ratio_10': 176073,
        'windowsize_28_stride_3_padding_2_negpos_ratio_1': 1061415,
        'windowsize_28_stride_3_padding_2_negpos_ratio_5': 3188079,
        'windowsize_28_stride_3_padding_2_negpos_ratio_10': 5846409,
        'windowsize_28_stride_3_padding_4_negpos_ratio_1': 738429,
        'windowsize_28_stride_3_padding_4_negpos_ratio_5': 2219129,
        'windowsize_28_stride_3_padding_4_negpos_ratio_10': 4070004,
        'windowsize_28_stride_3_padding_6_negpos_ratio_1': 475505,
        'windowsize_28_stride_3_padding_6_negpos_ratio_5': 1430377,
        'windowsize_28_stride_3_padding_6_negpos_ratio_10': 2623967,
        'windowsize_28_stride_3_padding_8_negpos_ratio_1': 267910,
        'windowsize_28_stride_3_padding_8_negpos_ratio_5': 807606,
        'windowsize_28_stride_3_padding_8_negpos_ratio_10': 1482226,
        'windowsize_28_stride_3_padding_10_negpos_ratio_1': 118784,
        'windowsize_28_stride_3_padding_10_negpos_ratio_5': 360232,
        'windowsize_28_stride_3_padding_10_negpos_ratio_10': 662042,
        'windowsize_28_stride_3_padding_12_negpos_ratio_1': 30425,
        'windowsize_28_stride_3_padding_12_negpos_ratio_5': 95169,
        'windowsize_28_stride_3_padding_12_negpos_ratio_10': 176099,
        'windowsize_32_stride_3_padding_2_negpos_ratio_1': 1440437,
        'windowsize_32_stride_3_padding_2_negpos_ratio_5': 4325145,
        'windowsize_32_stride_3_padding_2_negpos_ratio_10': 7931030,
        'windowsize_32_stride_3_padding_4_negpos_ratio_1': 1061967,
        'windowsize_32_stride_3_padding_4_negpos_ratio_5': 3189743,
        'windowsize_32_stride_3_padding_4_negpos_ratio_10': 5849463,
        'windowsize_32_stride_3_padding_6_negpos_ratio_1': 739791,
        'windowsize_32_stride_3_padding_6_negpos_ratio_5': 2223235,
        'windowsize_32_stride_3_padding_6_negpos_ratio_10': 4077540,
        'windowsize_32_stride_3_padding_8_negpos_ratio_1': 474596,
        'windowsize_32_stride_3_padding_8_negpos_ratio_5': 1427664,
        'windowsize_32_stride_3_padding_8_negpos_ratio_10': 2618999,
        'windowsize_32_stride_3_padding_10_negpos_ratio_1': 267344,
        'windowsize_32_stride_3_padding_10_negpos_ratio_5': 805912,
        'windowsize_32_stride_3_padding_10_negpos_ratio_10': 1479122,
        'windowsize_32_stride_3_padding_12_negpos_ratio_1': 118989,
        'windowsize_32_stride_3_padding_12_negpos_ratio_5': 360861,
        'windowsize_32_stride_3_padding_12_negpos_ratio_10': 663201,
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
        'windowsize_16_stride_3_padding_2': 1054563,
        'windowsize_16_stride_3_padding_4': 468348,
        'windowsize_16_stride_3_padding_6': 111673,
        'windowsize_20_stride_3_padding_2': 1878267,
        'windowsize_20_stride_3_padding_4': 1055226,
        'windowsize_20_stride_3_padding_6': 465207,
        'windowsize_20_stride_3_padding_8': 113081,
        'windowsize_24_stride_3_padding_2': 2915025,
        'windowsize_24_stride_3_padding_4': 1885606,
        'windowsize_24_stride_3_padding_6': 1054342,
        'windowsize_24_stride_3_padding_8': 461618,
        'windowsize_24_stride_3_padding_10': 114615,
        'windowsize_28_stride_3_padding_2': 4204332,
        'windowsize_28_stride_3_padding_4': 2932788,
        'windowsize_28_stride_3_padding_6': 1872983,
        'windowsize_28_stride_3_padding_8': 1053564,
        'windowsize_28_stride_3_padding_10': 466825,
        'windowsize_28_stride_3_padding_12': 111269,
        'windowsize_32_stride_3_padding_2': 5714430,
        'windowsize_32_stride_3_padding_4': 4208499,
        'windowsize_32_stride_3_padding_6': 2927364,
        'windowsize_32_stride_3_padding_8': 1877464,
        'windowsize_32_stride_3_padding_10': 1052073,
        'windowsize_32_stride_3_padding_12': 463205,
        'windowsize_32_stride_3_padding_14': 112617
    }

    localizer_valid_window_numbers = {
        'windowsize_16_stride_3_padding_2': 133077,
        'windowsize_16_stride_3_padding_4': 58563,
        'windowsize_16_stride_3_padding_6': 14268,
        'windowsize_20_stride_3_padding_2': 236503,
        'windowsize_20_stride_3_padding_4': 133155,
        'windowsize_20_stride_3_padding_6': 58754,
        'windowsize_20_stride_3_padding_8': 14192,
        'windowsize_24_stride_3_padding_2': 368069,
        'windowsize_24_stride_3_padding_4': 237038,
        'windowsize_24_stride_3_padding_6': 132905,
        'windowsize_24_stride_3_padding_8': 58538,
        'windowsize_24_stride_3_padding_10': 14243,
        'windowsize_28_stride_3_padding_2': 529749,
        'windowsize_28_stride_3_padding_4': 368254,
        'windowsize_28_stride_3_padding_6': 236787,
        'windowsize_28_stride_3_padding_8': 132986,
        'windowsize_28_stride_3_padding_10': 58422,
        'windowsize_28_stride_3_padding_12': 14239,
        'windowsize_32_stride_3_padding_2': 719260,
        'windowsize_32_stride_3_padding_4': 530023,
        'windowsize_32_stride_3_padding_6': 368930,
        'windowsize_32_stride_3_padding_8': 236329,
        'windowsize_32_stride_3_padding_10': 132702,
        'windowsize_32_stride_3_padding_12': 58521,
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
                        default=r'/raid/jsanders/seedNet2satNet',
                        help='Path to the directory containing all of the tfrecords files for the subwindows.')

    parser.add_argument('--window_size', type=str,
                        default='16,20,24,28,32',
                        help='Size of sub-windows (in pixels); single value or multiple values separated by a comma.')

    parser.add_argument('--stride', type=str,
                        default='3',
                        help='Stride of the sliding window (in pixels); single value or multiple values separated by a comma.')

    parser.add_argument('--padding', type=str,
                        default='2,4,6,8,10,12,14',
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
                        default=16384,
                        help='Number of images to prefetch in the input pipeline.')

    parser.add_argument('--batch_size', type=int,
                        default=7000,
                        help='Batch size to use in training and validation.')

    parser.add_argument('--num_dataset_threads', type=int,
                        default=4,
                        help='Number of threads to be used by the input pipeline.')

    parser.add_argument('--gpu_list', type=str,
                        default="3",
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
