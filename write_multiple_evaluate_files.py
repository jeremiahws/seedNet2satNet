

'''seedNet2satNet/write_multiple_evaluate_files.py

Evaluates multiple seedNet2satNet models and writes the inferences
to evaluate JSON files.
'''


import argparse
import os
import itertools
from utils.general_utils import ckpt2model


def main(FLAGS):
    paddings = FLAGS.padding.split(',')
    window_sizes = FLAGS.window_size.split(',')
    strides = FLAGS.stride.split(',')
    bg2sat_ratios = FLAGS.bg2sat_ratio.split(',')
    experiments = [window_sizes, strides, paddings, bg2sat_ratios]

    for experiment in itertools.product(*experiments):
        classifier_ckpt_name = 'classifier_seedNet2satNet_classifier_windowsize_{}_stride_{}_padding_{}_ratio_{}.h5'.format(experiment[0], experiment[1], experiment[2], experiment[3])
        localizer_ckpt_name = 'localizer_seedNet2satNet_localizer_windowsize_{}_stride_{}_padding_{}.h5'.format(experiment[0], experiment[1], experiment[2])
        classifier_model_name = 'model_classifier_seedNet2satNet_classifier_windowsize_{}_stride_{}_padding_{}_ratio_{}.h5'.format(experiment[0], experiment[1], experiment[2], experiment[3])
        localizer_model_name = 'model_localizer_seedNet2satNet_localizer_windowsize_{}_stride_{}_padding_{}.h5'.format(experiment[0], experiment[1], experiment[2])
        json_file = 'classifier_windowsize_{}_stride_{}_padding_{}_ratio_{}_localizer_windowsize_{}_stride_{}_padding_{}.json'.format(experiment[0], experiment[1], experiment[2], experiment[3],
                                                                                                                                      experiment[0], experiment[1], experiment[2])

        classifier_ckpt_path = os.path.join(FLAGS.classification_model_dir, classifier_ckpt_name)
        localizer_ckpt_path = os.path.join(FLAGS.localization_model_dir, localizer_ckpt_name)
        classifier_model_path = os.path.join(FLAGS.classification_model_dir, classifier_model_name)
        localizer_model_path = os.path.join(FLAGS.localization_model_dir, localizer_model_name)
        if FLAGS.ckpt2model:
            ckpt2model(classifier_ckpt_path, classifier_model_path)
            ckpt2model(localizer_ckpt_path, localizer_model_path)
            classifier_path = classifier_model_path
            localizer_path = localizer_model_path
        else:
            classifier_path = classifier_ckpt_path
            localizer_path = localizer_ckpt_path

        json_path = os.path.join(FLAGS.json_dir, json_file)
        create_command = 'python write_evaluate_file.py' \
                       + ' --window_size={}'.format(experiment[0])\
                       + ' --stride={}'.format(experiment[1])\
                       + ' --padding={}'.format(experiment[2])\
                       + ' --width={}'.format(FLAGS.width)\
                       + ' --height={}'.format(FLAGS.height)\
                       + ' --json_path={}'.format(json_path)\
                       + ' --classifier_path={}'.format(classifier_path)\
                       + ' --localizer_path={}'.format(localizer_path)\
                       + ' --test_file_names={}'.format(FLAGS.test_file_names)\
                       + ' --test_fraction={}'.format(FLAGS.test_fraction)\
                       + ' --n_test={}'.format(FLAGS.n_test)\
                       + ' --satnet_data_dir={}'.format(FLAGS.satnet_data_dir)\
                       + ' --batch_size={}'.format(FLAGS.batch_size)

        os.system(create_command)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

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

    parser.add_argument('--width', type=int,
                        default=512,
                        help='Width of the image (in pixels).')

    parser.add_argument('--height', type=int,
                        default=512,
                        help='Height of the image (in pixels).')

    parser.add_argument('--json_dir', type=str,
                        default='/home/jsanders/Desktop/github/seedNet2satNet/evaluate',
                        help='Path to directory where the JSON evaluate files should be saved.')

    parser.add_argument('--classification_model_dir', type=str,
                        default='/home/jsanders/Desktop/github/seedNet2satNet/classifiers',
                        help='Path to directory where classification models are stored.')

    parser.add_argument('--localization_model_dir', type=str,
                        default='/home/jsanders/Desktop/github/seedNet2satNet/localizers',
                        help='Path to directory where localization models are stored.')

    parser.add_argument('--test_file_names', type=str,
                        default='/opt/tfrecords/SatNet.v.1.1.0.0/SatNet/info/data_split/test.txt',
                        help='Path to .txt file containing testing file names.')

    parser.add_argument('--test_fraction', type=float,
                        default=0.05,
                        help='Fraction of total number of testing images to make predictions on.')

    parser.add_argument('--n_test', type=int,
                        default=10410,
                        help='Total number of SatNet testing images.')

    parser.add_argument('--satnet_data_dir', type=str,
                        default='/opt/tfrecords/SatNet.v.1.1.0.0/SatNet/data',
                        help='Top level directory for SatNet data from all sensors and collection days.')

    parser.add_argument('--batch_size', type=int,
                        default=7000,
                        help='Batch size to use in training and validation.')

    parser.add_argument('--gpu_list', type=str,
                        default="2",
                        help='GPUs to use with this model.')

    parser.add_argument('--ckpt2model', type=bool,
                        default=False,
                        help='The HDF5 files are checkpoints that need to be converted to models.')

    # parse known arguements
    FLAGS, unparsed = parser.parse_known_args()

    main(FLAGS)
