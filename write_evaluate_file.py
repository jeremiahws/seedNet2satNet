

'''seedNet2satNet/write_evaluate_file.py

Script to turn raw detection and classification inferences from seedNet2satNet
into bounding boxes around the detected objects.
'''


import argparse
from utils.image_reader import SatelliteImage
from utils.json_parser import ImageAnnotations
from utils.patch_extractor import SatNetSubWindows
from utils.single_inference import SeedNet2SatNetInference
from utils.general_utils import txt2list, ckpt2model
from keras.models import load_model
import tensorflow as tf
import json
from glob import glob
import os
import time
from math import floor
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(FLAGS):
    """Take sub-window classification and localization inferences and produce
    bounding boxes. The boxes are output to a .json file to be passed through
    the SatNet evaluate script.

    :param FLAGS: flags from the parser with inputs specified by the user
    :return: nothing
    """
    # create detections dictionary (based on Ian's format)
    detections_dict = {
        "image_name": [],
        "predicted_boxes": [],
        "predicted_scores": [],
        "ground_truth_boxes": [],
        "ground_truth_class_id": [],
    }

    # get the names of the test image files
    test_files = txt2list(FLAGS.test_file_names)

    # filtered images listed as JPEGs?? switch to FITS
    # comment out if using unfiltered data set
    test_files = [name.split('.jpg')[0] for name in test_files]
    test_files = [name + '.fits' for name in test_files]

    # set parameters for accumulator
    inference_limit = floor(FLAGS.test_fraction * FLAGS.n_test)
    inference_full = False

    # directories of sensor data and annotations
    sub_dirs = glob(os.path.join(FLAGS.satnet_data_dir, '*'))

    # load the seedNet2satNet models
    classifier_model_path = FLAGS.classifier_path.split('.h5')[0] + '_ckpt2model.h5'
    localizer_model_path = FLAGS.localizer_path.split('.h5')[0] + '_ckpt2model.h5'
    classifier = ckpt2model(FLAGS.classifier_path, classifier_model_path, return_model=True)
    localizer = ckpt2model(FLAGS.localizer_path, localizer_model_path, return_model=True)

    inference_count = 0
    inference_time = 0
    for dir in sub_dirs:
        img_files = glob(os.path.join(dir, 'ImageFiles', '*.fits'))
        json_files = glob(os.path.join(dir, 'Annotations', '*.json'))

        # get only the name of the .json file w/o extension
        json_names = [file.split("/")[-1] for file in json_files]
        json_names = [name.split(".json")[0] for name in json_names]
        # print(json_names)

        # get only the name of the .fits file w/o extension
        img_names = [file.split("/")[-1] for file in img_files]
        img_names = [name.split(".fits")[0] for name in img_names]
        # print(img_names)

        # in case some annotations/images aren't paired, find the
        # common .json and .fits files names
        similar_files = set(img_names).intersection(json_names)

        # prepare the new images and annotations via the sliding-window
        # algorithm
        for file in similar_files:
            # load SatNet image and its corresponding annotations
            img_path = os.path.join(dir, 'ImageFiles', file + '.fits')
            anno_path = os.path.join(dir, 'Annotations', file + '.json')
            image = SatelliteImage(img_path)
            anno = ImageAnnotations(anno_path)

            # find the data partition this example belongs to and add
            # that data to the accumulators
            comp_name = '_'.join([anno.directory, anno.name])

            if inference_count >= inference_limit:
                inference_full = True
                break

            if inference_full is False:
                if comp_name in test_files:
                    # start timer for single inference
                    tic = time.clock()

                    # pull all object centroids in the image and store in a list
                    gt_centroids = []
                    gt_boxes = []
                    [gt_centroids.append([obj.y_c, obj.x_c]) for obj in anno.objects]
                    [gt_boxes.append([obj.y_min, obj.x_min, obj.y_max, obj.x_max]) for obj in anno.objects]

                    # run sliding window algorithm across the image
                    sw = SatNetSubWindows(img=image.image,
                                          centroids=gt_centroids,
                                          window_size=FLAGS.window_size,
                                          stride=FLAGS.stride,
                                          padding=FLAGS.padding,
                                          img_width=FLAGS.width,
                                          img_height=FLAGS.height)

                    # normalize the sub-window intensities between [0, 1]
                    sw.z2o_normalize_windows(0.0, 65535.0)

                    # perform satellite detection inferences
                    inference_obj = SeedNet2SatNetInference(classifier, localizer, sw, gt_annos=anno, batch_size=7000)

                    # # apply clustering-based non-max suppression
                    # cluster_locs, cluster_scores = inference_obj.cluster_raw_detections(thresh=0.05)
                    # object_locs, object_boxes, object_scores = inference_obj.cluster_non_max_suppression(cluster_locs, cluster_scores, thresh=0.99)
                    # inference_obj.plot_final_preds(object_locs, plot_gt=True, plot_centroids=True)
                    # object_scores = [[1.0 - score, score] for score in object_scores]

                    if inference_obj.n_detections > 1:
                        # apply generic non-max suppression based on IoU
                        inds = inference_obj.non_max_suppression(inference_obj.raw_global_location_boxes,
                                                                 inference_obj.raw_pred_object_scores,
                                                                 conf_thresh=0.50, iou_thresh=0.0001, max_boxes=10)
                        with tf.Session() as sess:
                            sess.run(inds)
                            detection_inds = inds.eval()

                        object_locs = list(inference_obj.raw_global_location_preds[detection_inds])
                        object_boxes = [list(box) for box in inference_obj.raw_global_location_boxes[detection_inds]]
                        object_scores = list(inference_obj.raw_pred_object_scores[detection_inds])
                        object_scores = [[1.0 - score, score] for score in object_scores]

                    elif inference_obj.n_detections == 1:
                        object_locs = list(inference_obj.raw_global_location_preds)
                        object_boxes = list(inference_obj.raw_global_location_boxes)
                        object_scores = [1.0 - inference_obj.raw_pred_object_scores, inference_obj.raw_pred_object_scores]

                    else:
                        object_locs = []
                        object_boxes = []
                        object_scores = []

                    # plot the inferences
                    # inference_obj.plot_raw_inferences(plot_gt=True)
                    # inference_obj.plot_raw_boxes(plot_gt=True)
                    # inference_obj.plot_final_preds(object_locs, plot_gt=True, plot_centroids=True)

                    # stop timer for single inference
                    toc = time.clock()

                    # create the file name
                    file_name = '_'.join([anno.directory, anno.name])
                    parts = file_name.split('.fits')
                    file_name = parts[0]

                    # prepare class IDs to store
                    if gt_boxes:
                        class_ids = [1.0 for _ in range(len(gt_boxes))]
                    else:
                        class_ids = []

                    # add detections to the dictionary
                    detections_dict['image_name'].append(file_name)
                    detections_dict['predicted_boxes'].append(object_boxes)
                    detections_dict['predicted_scores'].append(object_scores)
                    detections_dict['ground_truth_boxes'].append(gt_boxes)
                    detections_dict['ground_truth_class_id'].append(class_ids)

                    inference_count += 1
                    inference_time += toc - tic
                    print('{} test inferences completed; {} total inference time'.format(inference_count, inference_time))

            else:
                pass

    with open(FLAGS.json_path, 'w') as f:
        json.dump(detections_dict, f, indent=1)

    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--window_size', type=int,
                        default=32,
                        help='Size of sub-windows (in pixels).')

    parser.add_argument('--stride', type=int,
                        default=3,
                        help='Stride of the sliding window (in pixels).')

    parser.add_argument('--padding', type=int,
                        default=12,
                        help='Padding to apply to sub-windows to avoid edge cases (in pixels).')

    parser.add_argument('--width', type=int,
                        default=512,
                        help='Width of the image (in pixels).')

    parser.add_argument('--height', type=int,
                        default=512,
                        help='Height of the image (in pixels).')

    parser.add_argument('--json_path', type=str,
                        default='/home/jsanders/Desktop/github/seedNet2satNet/window_32_stride_3_padding_12_ratio_10_trainfraction_1.0_tfnms.json',
                        help='Path to the JSON evaluate file to write.')

    parser.add_argument('--classifier_path', type=str,
                        default='/home/jsanders/Desktop/github/dlae_migration2/dlae/models/classification_ckpt_seedNet2satNet_windowsize_32_stride_3_padding_12_ratio_10_trainfraction_1.0.h5',
                        help='Path to the HDF5 file containing the seedNet2satNet classifier.')

    parser.add_argument('--localizer_path', type=str,
                        default='/home/jsanders/Desktop/github/dlae_migration2/dlae/models/localization_ckpt_seedNet2satNet_windowsize_32_stride_3_padding_12_ratio_10_trainfraction_1.0.h5',
                        help='Path to the HDF5 file containing the seedNet2satNet localizer.')

    parser.add_argument('--test_file_names', type=str,
                        default='/opt/tfrecords/SatNet.v.1.0.0.0/SatNet/info/data_split/test.txt',
                        help='Path to .txt file containing testing file names.')

    parser.add_argument('--test_fraction', type=float,
                        default=1.0,
                        help='Fraction of total number of testing images to make predictions on.')

    parser.add_argument('--n_test', type=int,
                        default=7572,
                        help='Total number of SatNet testing images.')

    parser.add_argument('--satnet_data_dir', type=str,
                        default='/opt/tfrecords/SatNet.v.1.0.0.0/SatNet/data',
                        help='Top level directory for SatNet data from all sensors and collection days.')

    # parse known arguements
    FLAGS, unparsed = parser.parse_known_args()

    main(FLAGS)
