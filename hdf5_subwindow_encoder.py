

"""seedNet2satNet/hdf5_subwindow_encoder.py

Encodes subwindows to an HDF5 file.
"""

import os
import numpy as np
import argparse
from utils.image_reader import SatelliteImage
from utils.json_parser import ImageAnnotations
from utils.patch_extractor import SatNetSubWindows
import h5py


def write_swcnn_hdf5(image_list, annotation_list, FLAGS, tvt):
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

    classification_hdf5_path1 = os.path.join(FLAGS.output_dir, path_append1 + '_classification.h5')
    localization_hdf5_path = os.path.join(FLAGS.output_dir, path_append_loc + '_localization.h5')
    classification_hdf5_path2 = os.path.join(FLAGS.output_dir, path_append2 + '_classification.h5')
    classification_hdf5_path3 = os.path.join(FLAGS.output_dir, path_append3 + '_classification.h5')

    # Open up TFRecord writers
    classification_writer1 = h5py.File(classification_hdf5_path1, 'w')
    localization_writer = h5py.File(localization_hdf5_path, 'w')
    classification_writer2 = h5py.File(classification_hdf5_path2, 'w')
    classification_writer3 = h5py.File(classification_hdf5_path3, 'w')

    # Write each example to file
    img_count = 0
    num_images = len(image_list)
    limit = int(FLAGS.fraction * num_images)
    for image_path, annotation_path in zip(image_list, annotation_list):
        if img_count > limit:
            break

        else:
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
                                  parallel=False,
                                  pad_img=False)
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

            windows1 = sw.windows_without[inds1]
            windows2 = sw.windows_without[inds2]
            windows3 = sw.windows_without[inds3]

            annos1 = np.zeros(windows1.shape[0])
            annos2 = np.zeros(windows2.shape[0])
            annos3 = np.zeros(windows3.shape[0])

            if sw.windows_with is not None:
                windows1 = np.concatenate((windows1, sw.windows_with), axis=0)
                windows2 = np.concatenate((windows2, sw.windows_with), axis=0)
                windows3 = np.concatenate((windows3, sw.windows_with), axis=0)
                annos1 = np.concatenate((annos1, np.ones(sw.windows_with.shape[0])))
                annos2 = np.concatenate((annos2, np.ones(sw.windows_with.shape[0])))
                annos3 = np.concatenate((annos3, np.ones(sw.windows_with.shape[0])))
                localization_writer.create_dataset(name=path_name + ':windows', data=sw.windows_with)
                localization_writer.create_dataset(name=path_name + ':annos', data=sw.object_location_with)

            classification_writer1.create_dataset(name=path_name + ':windows', data=windows1)
            classification_writer2.create_dataset(name=path_name + ':windows', data=windows2)
            classification_writer3.create_dataset(name=path_name + ':windows', data=windows3)
            classification_writer1.create_dataset(name=path_name + ':annos', data=annos1)
            classification_writer2.create_dataset(name=path_name + ':annos', data=annos2)
            classification_writer3.create_dataset(name=path_name + ':annos', data=annos3)

            img_count += 1

            if (img_count % 100) == 0:
                print("Images completed: {}".format(img_count))

    classification_writer1.close()
    classification_writer2.close()
    classification_writer3.close()
    localization_writer.close()


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
    # test_images, test_annotations = parse_file_list(FLAGS.test_file_names, FLAGS.data_dir)

    write_swcnn_hdf5(train_images, train_annotations, FLAGS, 'train')
    write_swcnn_hdf5(valid_images, valid_annotations, FLAGS, 'valid')
    # write_swcnn_hdf5(test_images, test_annotations, FLAGS, 'test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str,
                        # default="/opt/tfrecords/SatNet.v.1.1.0.0/SatNet/data",
                        default=r'C:\Users\jsanders\Desktop\data\seednet2satnet\SatNet_full_v2\SatNet\data',
                        help='Path to SatNet data directory.')

    parser.add_argument('--output_dir', type=str,
                        # default="/home/jsanders/Desktop/data",
                        default=r'C:\Users\jsanders\Desktop\Github\seedNet2satNet',
                        help='Path to the output directory for the tfrecords.')

    parser.add_argument('--train_file_names', type=str,
                        # default='/opt/tfrecords/SatNet.v.1.1.0.0/SatNet/info/data_split/train.txt',
                        default=r'C:\Users\jsanders\Desktop\data\seednet2satnet\SatNet_full_v2\SatNet\info\data_split\train.txt',
                        help='Path to .txt file containing training file names.')

    parser.add_argument('--valid_file_names', type=str,
                        # default='/opt/tfrecords/SatNet.v.1.1.0.0/SatNet/info/data_split/valid.txt',
                        default=r'C:\Users\jsanders\Desktop\data\seednet2satnet\SatNet_full_v2\SatNet\info\data_split\valid.txt',
                        help='Path to .txt file containing validation file names.')

    parser.add_argument('--test_file_names', type=str,
                        # default='/opt/tfrecords/SatNet.v.1.1.0.0/SatNet/info/data_split/test.txt',
                        default=r'C:\Users\jsanders\Desktop\data\seednet2satnet\SatNet_full_v2\SatNet\info\data_split\test.txt',
                        help='Path to .txt file containing testing file names.')

    parser.add_argument('--window_size', type=int,
                        default=24,
                        help='Size of sub-windows (in pixels).')

    parser.add_argument('--stride', type=int,
                        default=3,
                        help='Stride of the sliding window (in pixels).')

    parser.add_argument('--padding', type=int,
                        default=4,
                        help='Padding to apply to sub-windows to avoid edge cases (in pixels).')

    parser.add_argument('--fraction', type=float,
                        default=0.1,
                        help='Fraction of total number of images to write.')

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
