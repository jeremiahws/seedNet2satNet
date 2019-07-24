"""
TFRecord Builder
Author: 1st Lt Ian McQuaid
Date: 23 March 2019

Change log:
7/2/2019 - JWS - create sub-window tfrecords instead of image tfrecords.
    Two sets of tfrecords are made; one for the classifier, one for the
    localizer. Targets are sub-window classes and object centroids,
    respectively, instead of bounding boxes.
"""

import os
import tensorflow as tf
import numpy as np
import argparse
from utils.image_reader import SatelliteImage
from utils.json_parser import ImageAnnotations
from utils.patch_extractor import SatNetSubWindows
import csv


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def write_swcnn_tfrecord(image_list, annotation_list, FLAGS, tvt):
    bg2sat_ratios = FLAGS.bg2sat_ratio.split(',')
    ratio1 = int(bg2sat_ratios[0])
    ratio2 = int(bg2sat_ratios[1])
    ratio3 = int(bg2sat_ratios[2])
    path_append1 = 'seedNet2satNet_windowsize_{}_stride_{}_padding_{}_negposratio_{}_{}'.format(FLAGS.window_size,
                                                                                                FLAGS.stride,
                                                                                                FLAGS.padding,
                                                                                                ratio1,
                                                                                                tvt)

    path_append_loc = 'seedNet2satNet_windowsize_{}_stride_{}_padding_{}_{}'.format(FLAGS.window_size,
                                                                                    FLAGS.stride,
                                                                                    FLAGS.padding,
                                                                                    tvt)

    path_append2 = 'seedNet2satNet_windowsize_{}_stride_{}_padding_{}_negposratio_{}_{}'.format(FLAGS.window_size,
                                                                                                FLAGS.stride,
                                                                                                FLAGS.padding,
                                                                                                ratio2,
                                                                                                tvt)

    path_append3 = 'seedNet2satNet_windowsize_{}_stride_{}_padding_{}_negposratio_{}_{}'.format(FLAGS.window_size,
                                                                                                FLAGS.stride,
                                                                                                FLAGS.padding,
                                                                                                ratio3,
                                                                                                tvt)

    classification_tfrecord_path1 = os.path.join(FLAGS.output_dir, path_append1 + '_classification.tfrecords')
    localization_tfrecord_path = os.path.join(FLAGS.output_dir, path_append_loc + '_localization.tfrecords')
    classification_tfrecord_path2 = os.path.join(FLAGS.output_dir, path_append2 + '_classification.tfrecords')
    classification_tfrecord_path3 = os.path.join(FLAGS.output_dir, path_append3 + '_classification.tfrecords')

    # Open up TFRecord writers
    classification_writer1 = tf.python_io.TFRecordWriter(classification_tfrecord_path1)
    localization_writer = tf.python_io.TFRecordWriter(localization_tfrecord_path)
    classification_writer2 = tf.python_io.TFRecordWriter(classification_tfrecord_path2)
    classification_writer3 = tf.python_io.TFRecordWriter(classification_tfrecord_path3)

    # Write each example to file
    img_count = 0
    c_window_count1 = 0
    l_window_count = 0
    c_window_count2 = 0
    c_window_count3 = 0
    num_images = len(image_list)
    for image_path, annotation_path in zip(image_list, annotation_list):
        if (img_count % 100) == 0:
            print("Image {}/{}, C windows ratio 1: {}, L windows: {}".format(img_count,
                                                                             num_images,
                                                                             c_window_count1,
                                                                             l_window_count))
            print("Image {}/{}, C windows ratio 2: {}".format(img_count,
                                                              num_images,
                                                              c_window_count2))
            print("Image {}/{}, C windows ratio 3: {}".format(img_count,
                                                              num_images,
                                                              c_window_count3))

        image = SatelliteImage(image_path)
        anno = ImageAnnotations(annotation_path)

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
                              img_height=FLAGS.height,
                              parallel=False)
        sw.get_obj_windows()

        # find how many background windows to include from the image and generate that many number of random indices
        # to pull them
        if sw.windows_with is not None:
            n_with = sw.windows_with.shape[0]
            n_without1 = int(ratio1 * n_with)
            n_without2 = int(ratio2 * n_with)
            n_without3 = int(ratio3 * n_with)
        else:
            n_without1 = ratio1
            n_without2 = ratio2
            n_without3 = ratio3
        inds = np.random.permutation(sw.windows_without.shape[0])
        inds1 = inds[:n_without1]
        inds2 = inds[:n_without2]
        inds3 = inds[:n_without3]

        dir_name = anno.directory
        file_name = anno.name

        # Remove the extension (to be compatible with what Greg did...)
        file_name = file_name.replace('.fits', '')

        # Put the two together via '_' (again, to be compatible...)
        path_name = dir_name + "_" + file_name

        if sw.windows_with is not None:
            for i, window in enumerate(sw.windows_with):
                # classification features
                classification_feature = {
                    "filename": _bytes_feature([path_name.encode()]),
                    "window_height": _int64_feature([sw.window_size]),
                    "window_width": _int64_feature([sw.window_size]),
                    "window": _bytes_feature([window.tostring()]),
                    "annotation": _int64_feature([1])
                }

                # Create an example protocol buffer
                classification_example = tf.train.Example(features=tf.train.Features(feature=classification_feature))
                # Serialize to string and write on the file
                classification_writer1.write(classification_example.SerializeToString())
                c_window_count1 += 1
                classification_writer2.write(classification_example.SerializeToString())
                c_window_count2 += 1
                classification_writer3.write(classification_example.SerializeToString())
                c_window_count3 += 1

                # localization features
                localization_feature = {
                    "filename": _bytes_feature([path_name.encode()]),
                    "window_height": _int64_feature([sw.window_size]),
                    "window_width": _int64_feature([sw.window_size]),
                    "window": _bytes_feature([window.tostring()]),
                    "y_c": _floats_feature([sw.object_location_with[i, 0]]),
                    "x_c": _floats_feature([sw.object_location_with[i, 1]])
                }

                # Create an example protocol buffer
                localization_example = tf.train.Example(features=tf.train.Features(feature=localization_feature))
                # Serialize to string and write on the file
                localization_writer.write(localization_example.SerializeToString())
                l_window_count += 1

        for i, window in enumerate(sw.windows_without[inds1]):
            # classification features
            classification_feature = {
                "filename": _bytes_feature([path_name.encode()]),
                "window_height": _int64_feature([sw.window_size]),
                "window_width": _int64_feature([sw.window_size]),
                "window": _bytes_feature([window.tostring()]),
                "annotation": _int64_feature([0])
            }

            # Create an example protocol buffer
            classification_example = tf.train.Example(features=tf.train.Features(feature=classification_feature))
            # Serialize to string and write on the file
            classification_writer1.write(classification_example.SerializeToString())
            c_window_count1 += 1

        for i, window in enumerate(sw.windows_without[inds2]):
            # classification features
            classification_feature = {
                "filename": _bytes_feature([path_name.encode()]),
                "window_height": _int64_feature([sw.window_size]),
                "window_width": _int64_feature([sw.window_size]),
                "window": _bytes_feature([window.tostring()]),
                "annotation": _int64_feature([0])
            }

            # Create an example protocol buffer
            classification_example = tf.train.Example(features=tf.train.Features(feature=classification_feature))
            # Serialize to string and write on the file
            classification_writer2.write(classification_example.SerializeToString())
            c_window_count2 += 1

        for i, window in enumerate(sw.windows_without[inds3]):
            # classification features
            classification_feature = {
                "filename": _bytes_feature([path_name.encode()]),
                "window_height": _int64_feature([sw.window_size]),
                "window_width": _int64_feature([sw.window_size]),
                "window": _bytes_feature([window.tostring()]),
                "annotation": _int64_feature([0])
            }

            # Create an example protocol buffer
            classification_example = tf.train.Example(features=tf.train.Features(feature=classification_feature))
            # Serialize to string and write on the file
            classification_writer3.write(classification_example.SerializeToString())
            c_window_count3 += 1

        img_count += 1

        print("Total numbers of windows: {} C1, {} C2, {} C3, {} L".format(c_window_count1,
                                                                           c_window_count2,
                                                                           c_window_count3,
                                                                           l_window_count))

    with open('windowsize_{}_stride_{}_padding_{}_negposratio_{}_window_count_{}.csv'.format(FLAGS.window_size,
                                                                                             FLAGS.stride,
                                                                                             FLAGS.padding,
                                                                                             ratio1,
                                                                                             tvt),
              'w', newline='') as f:
        fieldnames = ['classification', 'localization']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow({'classification': c_window_count1, 'localization': l_window_count})

    with open('windowsize_{}_stride_{}_padding_{}_negposratio_{}_window_count_{}.csv'.format(FLAGS.window_size,
                                                                                             FLAGS.stride,
                                                                                             FLAGS.padding,
                                                                                             ratio2,
                                                                                             tvt),
              'w', newline='') as f:
        fieldnames = ['classification', 'localization']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow({'classification': c_window_count2, 'localization': l_window_count})

    with open('windowsize_{}_stride_{}_padding_{}_negposratio_{}_window_count_{}.csv'.format(FLAGS.window_size,
                                                                                             FLAGS.stride,
                                                                                             FLAGS.padding,
                                                                                             ratio3,
                                                                                             tvt),
              'w', newline='') as f:
        fieldnames = ['classification', 'localization']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow({'classification': c_window_count3, 'localization': l_window_count})


def parse_file_list(split_path, data_path):
    # Read the split file in
    fp = open(split_path, "r")
    file_contents = fp.read()
    fp.close()

    # Split by line break
    file_list = file_contents.split("\n")

    # Remove the extension (we will add them back in later)
    file_list = [".".join(name.split(".")[:-1]) for name in file_list]

    # Split into the collect name and file name (because string manipulation is fun!)
    collect_list = [name.split("_")[0] for name in file_list]
    file_list = ["_".join(name.split("_")[1:]) for name in file_list]

    image_list = []
    annotation_list = []
    for i in range(len(collect_list)):
        collect_path = collect_list[i]
        file_name = file_list[i]
        if len(file_name) > 0:
            # Build the path to each .fits image
            img_path = os.path.join(data_path, collect_path, "ImageFiles", file_name + ".fits")
            annotation_path = os.path.join(data_path, collect_path, "Annotations", file_name + ".json")
            image_list.append(img_path)
            annotation_list.append(annotation_path)

    # Make a list for images and for annotations
    return image_list, annotation_list


def main(FLAGS):
    # Read in the files that we need in each individual TFRecord
    train_images, train_annotations = parse_file_list(FLAGS.train_file_names, FLAGS.data_dir)
    valid_images, valid_annotations = parse_file_list(FLAGS.valid_file_names, FLAGS.data_dir)
    test_images, test_annotations = parse_file_list(FLAGS.test_file_names, FLAGS.data_dir)

    write_swcnn_tfrecord(train_images, train_annotations, FLAGS, 'train')
    write_swcnn_tfrecord(valid_images, valid_annotations, FLAGS, 'valid')
    # write_swcnn_tfrecord(test_images, test_annotations, FLAGS, 'test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str,
                        default="/opt/tfrecords/SatNet.v.1.1.0.0/SatNet/data",
                        help='Path to SatNet data directory.')

    parser.add_argument('--output_dir', type=str,
                        default="/raid/jsanders/seedNet2satNet",
                        help='Path to the output directory for the tfrecords.')

    parser.add_argument('--train_file_names', type=str,
                        default='/opt/tfrecords/SatNet.v.1.1.0.0/SatNet/info/data_split/train.txt',
                        help='Path to .txt file containing training file names.')

    parser.add_argument('--valid_file_names', type=str,
                        default='/opt/tfrecords/SatNet.v.1.1.0.0/SatNet/info/data_split/valid.txt',
                        help='Path to .txt file containing validation file names.')

    parser.add_argument('--test_file_names', type=str,
                        default='/opt/tfrecords/SatNet.v.1.1.0.0/SatNet/info/data_split/test.txt',
                        help='Path to .txt file containing testing file names.')

    parser.add_argument('--window_size', type=int,
                        default=24,
                        help='Size of sub-windows (in pixels).')

    parser.add_argument('--stride', type=int,
                        default=1,
                        help='Stride of the sliding window (in pixels).')

    parser.add_argument('--padding', type=int,
                        default=4,
                        help='Padding to apply to sub-windows to avoid edge cases (in pixels).')

    parser.add_argument('--bg2sat_ratio', type=str,
                        default='1,5,10',
                        help='Ratio of background:satellite sub-windows in the training dataset.'
                             + ' Expects three values separated by a comma')

    parser.add_argument('--width', type=int,
                        default=512,
                        help='Width of the image (in pixels).')

    parser.add_argument('--height', type=int,
                        default=512,
                        help='Height of the image (in pixels).')

    FLAGS, unparsed = parser.parse_known_args()

    main(FLAGS)

# w_s=24, s=3, p=4, npr=10 train_c: 20938046 train_l: 1889036
#                          valid_c: 2630327  valid_l: 237367
#                           test_c: 2635187   test_l: 237797
