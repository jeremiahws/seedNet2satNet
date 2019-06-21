

'''seedNet2satNet/create_annotated_windows.py

Script to pair .fits images with object annotations in the sub-window fashion.
'''


from glob import glob
import argparse
from utils.image_reader import SatelliteImage
from utils.json_parser import ImageAnnotations
from utils.patch_extractor import SatNetSubWindows
from utils.general_utils import txt2list, write_hdf5
from math import floor
import numpy as np
import os


def main(FLAGS):
    """Curate sub-windows for seedNet2satNet experimentation.

    :param FLAGS: flags from the parser with inputs specified by the user
    :return: nothing
    """
    if FLAGS.format == 'tfrecords':
        raise NotImplementedError
    else:
        # get the names of the train image files
        train_files = txt2list(FLAGS.train_file_names)
        train_limit = floor(FLAGS.train_fraction * FLAGS.n_train)
        train_count = 0
        train_full = False

        # get the names of the validation image files
        valid_files = txt2list(FLAGS.valid_file_names)
        valid_limit = floor(FLAGS.valid_fraction * FLAGS.n_valid)
        valid_count = 0
        valid_full = False

        # get the names of the test image files
        test_files = txt2list(FLAGS.test_file_names)
        test_limit = floor(FLAGS.test_fraction * FLAGS.n_test)
        test_count = 0
        test_full = False

        # accumulators for the image and annotation pairs
        train_windows_with = []
        valid_windows_with = []
        test_windows_with = []
        train_windows_without = []
        valid_windows_without = []
        test_windows_without = []
        train_locations = []
        valid_locations = []
        test_locations = []

        # directories of sensor data and annotations
        sub_dirs = glob(os.path.join(FLAGS.satnet_data_dir, '*'))

        # go through each sensor collection from each site and prepare
        # the training, validation, and testing sub-windows
        for dir in sub_dirs:
            if train_full and valid_full and test_full:
                pass
            else:
                img_files = glob(os.path.join(dir, 'ImageFiles', '*.fits'))
                json_files = glob(os.path.join(dir, 'Annotations', '*.json'))

                # get only the name of the .json file w/o extension
                json_names = [file.split("\\")[-1] for file in json_files]
                json_names = [name.split(".json")[0] for name in json_names]

                # get only the name of the .fits file w/o extension
                img_names = [file.split("\\")[-1] for file in img_files]
                img_names = [name.split(".fits")[0] for name in img_names]

                # in case some annotations/images aren't paired, find the
                # common .json and .fits files names
                similar_files = set(img_names).intersection(json_names)

                # prepare the new images and annotations via the sliding-window
                # algorithm
                for file in similar_files:
                    if train_full and valid_full and test_full:
                        pass
                    else:
                        # load SatNet image and its corresponding annotations
                        img_path = os.path.join(dir, 'ImageFiles', file + '.fits')
                        anno_path = os.path.join(dir, 'Annotations', file + '.json')
                        image = SatelliteImage(img_path)
                        anno = ImageAnnotations(anno_path)

                        # find the data partition this example belongs to and add
                        # that data to the accumulators
                        comp_name = '_'.join([anno.directory, anno.name])

                        # pull all object centroids in the image and store in a list
                        centroids = []
                        [centroids.append([obj.y_c, obj.x_c]) for obj in anno.objects]

                        # run sliding window algorithm across the image
                        sw = SatNetSubWindows(img=image.image,
                                              centroids=centroids,
                                              window_size=FLAGS.window_size,
                                              stride=FLAGS.stride,
                                              padding=FLAGS.padding,
                                              img_width=FLAGS.width,
                                              img_height=FLAGS.height)
                        sw.get_obj_windows()

                        # find how many background windows to include from the image
                        # and generate that many number of random indices to pull
                        # them
                        if sw.windows_with is not None:
                            n_with = sw.windows_with.shape[0]
                            n_without = int(FLAGS.bg2sat_ratio * n_with)
                        else:
                            n_without = int(FLAGS.bg2sat_ratio)
                        inds = np.random.permutation(sw.windows_without.shape[0])
                        inds = inds[:n_without]

                        # determine the status of the accumulators
                        if train_count >= train_limit:
                            train_full = True
                        if valid_count >= valid_limit:
                            valid_full = True
                        if test_count >= test_limit:
                            test_full = True

                        # accumulate sub-windows into the three data
                        # partitions
                        if comp_name in train_files and not train_full:
                            if sw.windows_with is not None:
                                train_windows_with.append(sw.windows_with)
                                train_locations.append(sw.object_location_with)
                            train_windows_without.append(sw.windows_without[inds, :, :])
                            train_count += 1
                        elif comp_name in valid_files and not valid_full:
                            if sw.windows_with is not None:
                                valid_windows_with.append(sw.windows_with)
                                valid_locations.append(sw.object_location_with)
                            valid_windows_without.append(sw.windows_without[inds, :, :])
                            valid_count += 1
                        elif comp_name in test_files and not test_full:
                            if sw.windows_with is not None:
                                test_windows_with.append(sw.windows_with)
                                test_locations.append(sw.object_location_with)
                            test_windows_without.append(sw.windows_without[inds, :, :])
                            test_count += 1
                        else:
                            print('Windows belong to a filled accumulator... skipped them.')
                            pass
                        print('Accumulators: train - {}% , valid - {}% , test - {}%'.format(
                            int(train_count / train_limit * 100),
                            int(valid_count / valid_limit * 100),
                            int(test_count / test_limit * 100)))

        # combine all of the sub-windows and annotations for each data
        # partition
        train_windows_with = np.concatenate(train_windows_with)
        train_windows_without = np.concatenate(train_windows_without)
        train_locations = np.concatenate(train_locations)
        train_annos_with = np.ones(train_windows_with.shape[0])
        train_annos_without = np.zeros(train_windows_without.shape[0])
        valid_windows_with = np.concatenate(valid_windows_with)
        valid_windows_without = np.concatenate(valid_windows_without)
        valid_locations = np.concatenate(valid_locations)
        valid_annos_with = np.ones(valid_windows_with.shape[0])
        valid_annos_without = np.zeros(valid_windows_without.shape[0])
        test_windows_with = np.concatenate(test_windows_with)
        test_windows_without = np.concatenate(test_windows_without)
        test_locations = np.concatenate(test_locations)
        test_annos_with = np.ones(test_windows_with.shape[0])
        test_annos_without = np.zeros(test_windows_without.shape[0])

        train_windows = np.concatenate((train_windows_with, train_windows_without))
        train_annos = np.concatenate((train_annos_with, train_annos_without))
        valid_windows = np.concatenate((valid_windows_with, valid_windows_without))
        valid_annos = np.concatenate((valid_annos_with, valid_annos_without))
        test_windows = np.concatenate((test_windows_with, test_windows_without))
        test_annos = np.concatenate((test_annos_with, test_annos_without))

        path_append = '_seedNet2satNet_windowsize_{}_stride_{}_padding_{}_ratio_{}_trainfraction_{}.h5'.format(FLAGS.window_size, FLAGS.stride, FLAGS.padding, FLAGS.bg2sat_ratio, FLAGS.train_fraction)
        train_c_windows_path = os.path.join(FLAGS.save_data_dir, 'train_classification_windows' + path_append)
        train_c_labels_path = os.path.join(FLAGS.save_data_dir, 'train_classification_labels' + path_append)
        train_l_windows_path = os.path.join(FLAGS.save_data_dir, 'train_localization_windows' + path_append)
        train_l_labels_path = os.path.join(FLAGS.save_data_dir, 'train_localization_labels' + path_append)
        valid_c_windows_path = os.path.join(FLAGS.save_data_dir, 'valid_classification_windows' + path_append)
        valid_c_labels_path = os.path.join(FLAGS.save_data_dir, 'valid_classification_labels' + path_append)
        valid_l_windows_path = os.path.join(FLAGS.save_data_dir, 'valid_localization_windows' + path_append)
        valid_l_labels_path = os.path.join(FLAGS.save_data_dir, 'valid_localization_labels' + path_append)
        test_c_windows_path = os.path.join(FLAGS.save_data_dir, 'test_classification_windows' + path_append)
        test_c_labels_path = os.path.join(FLAGS.save_data_dir, 'test_classification_labels' + path_append)
        test_l_windows_path = os.path.join(FLAGS.save_data_dir, 'test_localization_windows' + path_append)
        test_l_labels_path = os.path.join(FLAGS.save_data_dir, 'test_localization_labels' + path_append)

        write_hdf5(train_c_windows_path, train_windows)
        write_hdf5(train_c_labels_path, train_annos)
        write_hdf5(train_l_windows_path, train_windows_with)
        write_hdf5(train_l_labels_path, train_locations)
        write_hdf5(valid_c_windows_path, valid_windows)
        write_hdf5(valid_c_labels_path, valid_annos)
        write_hdf5(valid_l_windows_path, valid_windows_with)
        write_hdf5(valid_l_labels_path, valid_locations)
        write_hdf5(test_c_windows_path, test_windows)
        write_hdf5(test_c_labels_path, test_annos)
        write_hdf5(test_l_windows_path, test_windows_with)
        write_hdf5(test_l_labels_path, test_locations)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--satnet_data_dir', type=str,
                        default='C:/Users/jsanders/Desktop/data/seednet2satnet/SatNet_full/SatNet/data',
                        help='Top level directory for SatNet data from all sensors and collection days.')

    parser.add_argument('--save_data_dir', type=str,
                        default='C:/Users/jsanders/Desktop/data/seedNet2satNet/SatNet_full/SatNet/data',
                        help='Directory where to save the sub-window data.')

    parser.add_argument('--train_file_names', type=str,
                        default='C:/Users/jsanders/Desktop/data/seedNet2satNet/SatNet_full/SatNet/info/data_split/train.txt',
                        help='Path to .txt file containing training file names.')

    parser.add_argument('--valid_file_names', type=str,
                        default='C:/Users/jsanders/Desktop/data/seedNet2satNet/SatNet_full/SatNet/info/data_split/valid.txt',
                        help='Path to .txt file containing validation file names.')

    parser.add_argument('--test_file_names', type=str,
                        default='C:/Users/jsanders/Desktop/data/seedNet2satNet/SatNet_full/SatNet/info/data_split/test.txt',
                        help='Path to .txt file containing testing file names.')

    parser.add_argument('--train_fraction', type=float,
                        default=1.0,
                        help='Fraction of total number of training images to curate.')

    parser.add_argument('--valid_fraction', type=float,
                        default=1.0,
                        help='Fraction of total number of validation images to curate.')

    parser.add_argument('--test_fraction', type=float,
                        default=1.0,
                        help='Fraction of total number of testing images to curate.')

    parser.add_argument('--n_train', type=int,
                        default=83256,
                        help='Total number of SatNet training images.')

    parser.add_argument('--n_valid', type=int,
                        default=10410,
                        help='Total number of SatNet validation images.')

    parser.add_argument('--n_test', type=int,
                        default=10410,
                        help='Total number of SatNet testing images.')

    parser.add_argument('--window_size', type=int,
                        default=32,
                        help='Size of sub-windows (in pixels).')

    parser.add_argument('--stride', type=int,
                        default=3,
                        help='Stride of the sliding window (in pixels).')

    parser.add_argument('--padding', type=int,
                        default=4,
                        help='Padding to apply to sub-windows to avoid edge cases (in pixels).')

    parser.add_argument('--width', type=int,
                        default=512,
                        help='Width of the image (in pixels).')

    parser.add_argument('--height', type=int,
                        default=512,
                        help='Height of the image (in pixels).')

    parser.add_argument('--bg2sat_ratio', type=int,
                        default=10,
                        help='Ratio of background:satellite sub-windows in the training dataset.')

    parser.add_argument('--format', type=str,
                        default='hdf5',
                        help='File format to save images and annotations in (hdf or tfrecords).')

    # parse known arguements
    FLAGS, unparsed = parser.parse_known_args()

    main(FLAGS)
