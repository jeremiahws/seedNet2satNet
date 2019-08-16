

'''seedNet2satNet/utils/single_inference.py

Performs inference on a SatNet image.
'''


import numpy as np
from operator import add
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import scipy.cluster.hierarchy as hcluster
import tensorflow as tf
from utils.image_reader import SatelliteImage
from utils.json_parser import ImageAnnotations
from utils.patch_extractor import SatNetSubWindows
from tensorflow.keras.models import load_model


def crop_image(img, cropx, cropy):
    """Crop center portion of an image.

    :param img: original image
    :param cropx: amount to crop in width direction
    :param cropy: amount to crop in height direction
    :return: cropped image
    """
    y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)

    return img[starty:starty + cropy, startx:startx + cropx]


class SeedNet2SatNetInference(object):
    """Make predictions on a single SatNet image.

        Attributes:
            sliding_window (object): the sliding-window object of the image
            box_size (int): size of both dimensions of the bounding boxes
                (in pixels)
            gt_annos (object): the ground truth annotations object for the
                image
            raw_class_preds (float): raw inferences of the sub-window class,
                before a non-max suppression is applied. Values are in the
                range [0, 1]
            raw_location_preds (float): raw inferences of the satellite
                location within the sub-window. Values are of a fraction of
                the original image width/height and are relative to the top
                left corner of the sub-window. These are in the format
                (y_c, x_c)
            raw_global_location_preds (float): raw inferences of the satellite
                location within the image. Values are of a fraction of
                the original image width/height and are relative to the top
                left corner of the image
            raw_global_location_boxes (float): bounding boxes around the raw
                object centroids predicted from the seedNet2satNet localizer.
                Values are of a fraction of the image width/height and are
                relative to the top left corner of the image. These are in the
                format (ymin, xmin, ymax, xmax)
            class_preds (int): sub-window class obtained by doing an argmax
                on the raw sub-window classification prediction
            n_detections (int): total number of sub-windows detected as having
                a satellite contained within the sub-window, before a non-max
                suppression is applied
            satellite_class_preds (float): class value of the detected
                satellites required for the evaluate script (will always be
                1.0 for single object class detection). These are stored
                before a non-max suppression is applied
            raw_pred_object_scores (float): raw classification inferences for
                the windows classified as satellite. Useful for the non-max
                suppression
            satellite_window_preds (float): sub-windows determined as having a
                satellite contained within the window, before a non-max
                suppression is applied
            satellite_window_gt (float): ground truth sub-windows containing a
                satellite. These sub-windows correspond to the sub-windows
                that were detected, whether or not they actually contain a
                satellite (i.e. FP). These can be useful to visually compare
                TP and FP sub-windows
            satellite_window_gt_location (float): ground truth location of the
                satellite within the sub-window relative to the top left
                corner of the sub-window. These values correspond to the
                sub-windows that were detected, whether or not they actually
                contain a satellite (i.e. FP, in which case the location is
                [0.0, 0.0])
            satellite_window_corner_coords (float): corner coordinates of the
                sub-windows classified as containing a satellite relative to
                the top left corner of the image. These values are used to map
                the sub-windows back to the image reference frame
    """
    def __init__(self,
                 classification_model,
                 localization_model,
                 sliding_window,
                 batch_size=64,
                 box_size=20,
                 padding=0,
                 gt_annos=None):
        """Performs inference on a SatNet image upon initialization.

        :param classification_model: the trained seedNet2satNet sub-window
                classification model already loaded as a keras model
        :param localization_model: the trained seedNet2satNet sub-window
                localization model already loaded as a keras model
        :param sliding_window: a sliding-window object that has already
                been processed by the sliding-window algorithm
        :param batch_size: the predictions batch size
        :param box_size: the size of the bounding box used to encompass the
            detected satellite
        :param gt_annos: ground truth annotations object
        """
        self.box_size = box_size
        self.gt_annos = gt_annos
        self.sliding_window = sliding_window
        self.raw_class_preds = classification_model.predict(sliding_window.windows, batch_size=batch_size)
        self.class_preds = np.argmax(self.raw_class_preds, axis=1)

        inds = np.where(self.class_preds == 1)
        self.n_detections = len(inds[0])
        self.satellite_class_preds = self.class_preds[inds].astype(float)
        self.raw_pred_object_scores = np.squeeze(self.raw_class_preds[inds, 1])
        self.satellite_window_preds = sliding_window.windows[inds]
        self.satellite_window_gt = sliding_window.object_present[inds]
        self.satellite_window_gt_location = sliding_window.object_location[inds]
        self.satellite_window_corner_coords = sliding_window.window_corner_coords[inds]

        self.raw_location_preds = []
        self.raw_global_location_preds = []
        self.raw_global_location_boxes = []
        if self.n_detections > 0:
            self.raw_location_preds = localization_model.predict(self.satellite_window_preds, batch_size=batch_size)

            x_delta = 0.5 * self.box_size / self.sliding_window.img_width
            y_delta = 0.5 * self.box_size / self.sliding_window.img_height
            for i, location in enumerate(self.raw_location_preds):
                self.raw_global_location_preds.append(list(map(add, location, self.satellite_window_corner_coords[i])))

                self.raw_global_location_boxes.append(
                    [(self.raw_global_location_preds[i][0] - y_delta),
                     (self.raw_global_location_preds[i][1] - x_delta),
                     (self.raw_global_location_preds[i][0] + y_delta),
                     (self.raw_global_location_preds[i][1] + x_delta)])

            self.raw_global_location_preds = np.asarray(self.raw_global_location_preds)
            self.raw_global_location_boxes = np.asarray(self.raw_global_location_boxes)

    def plot_raw_inferences(self, plot_gt=False, conf_thresh=None):
        """Plot the image with the inferred object locations without non-max
        suppression.

        :param plot_gt: flag to plot ground truth annotations with the raw
            inferences
        :return: nothing
        """
        plt.imshow(self.sliding_window.image, cmap='gray')
        if self.n_detections > 0:
            x_c = []
            y_c = []
            for location in self.raw_global_location_preds:
                x_c.append(location[1] * self.sliding_window.img_width)
                y_c.append(location[0] * self.sliding_window.img_height)

            plt.scatter(x=x_c, y=y_c, c='r', marker='x', s=30)
        else:
            pass

        if plot_gt:
            if self.gt_annos is None:
                pass
            else:
                if any(self.gt_annos.objects):
                    for obj in self.gt_annos.objects:
                        plt.scatter(x=[obj.x_c * self.sliding_window.img_width],
                                    y=[obj.y_c * self.sliding_window.img_height],
                                    c='y', marker='x', s=30)

        plt.show()

        return

    def plot_raw_boxes(self, plot_gt=False):
        """Plot the image with bounding boxes computed around the inferred
         object locations without non-max suppression.

        :param plot_gt: flag to plot ground truth bounding boxes with the
            raw bounding boxes
        :return: nothing
        """
        plt.imshow(self.sliding_window.image, cmap='gray')
        if self.n_detections > 0:
            for location in self.raw_global_location_boxes:
                x_min = location[1] * self.sliding_window.img_width
                y_min = location[0] * self.sliding_window.img_height
                plt.gca().add_patch(Rectangle((x_min, y_min),
                                              self.box_size, self.box_size,
                                              linewidth=1,
                                              edgecolor='r',
                                              facecolor='none'))

        else:
            pass

        if plot_gt:
            if self.gt_annos is None:
                pass
            else:
                if any(self.gt_annos.objects):
                    for obj in self.gt_annos.objects:
                        x_min = obj.x_min * self.sliding_window.img_width
                        y_min = obj.y_min * self.sliding_window.img_height
                        plt.gca().add_patch(Rectangle((x_min, y_min),
                                                      self.box_size, self.box_size,
                                                      linewidth=1,
                                                      edgecolor='y',
                                                      facecolor='none'))

        plt.show()

        return

    def pred_locations_to_mask(self):
        """Convert predicted object locations to a binary mask where pixels
        where an object was detected are 1's and background are 0's.

        :return: mask of predicted satellite locations
        """
        mask = np.zeros([self.sliding_window.img_height, self.sliding_window.img_width])
        if self.n_detections > 0:
            for location in self.raw_global_location_preds:
                mask[int(round(location[0] * self.sliding_window.img_height)),
                     int(round(location[1] * self.sliding_window.img_width))] = 1
        else:
            pass

        return mask

    def cluster_raw_detections(self, thresh):
        """Cluster the global raw location detections using hierarchical
        clustering based on Euclidean distance.

        :param thresh: threshold for splitting detections (clusters)
        :return: coordinates and raw class predictions of the clustered
            objects
        """
        cluster_locations = []
        cluster_satellite_preds = []
        if self.n_detections > 1:
            clusters = hcluster.fclusterdata(self.raw_global_location_preds, thresh, criterion="distance")
            cluster_nums = np.unique(clusters)
            for c in cluster_nums:
                inds = np.where(clusters == c)
                cluster_locations.append(self.raw_global_location_preds[inds])
                cluster_satellite_preds.append(self.raw_pred_object_scores[inds])
        elif self.n_detections == 1:
            cluster_locations.append(self.raw_global_location_preds)
            cluster_satellite_preds.append(self.raw_pred_object_scores)
        else:
            pass

        return cluster_locations, cluster_satellite_preds

    def radial_non_max_supression(self, locations, scores, radius):
        """Custom non-max suppression for the seedNet2satNet methodology
        using a radial location threshold.

        :param locations: list of location clusters around each detection
        :param scores: list of object scores around each detection
        :param radius: the radius to search for non-max suppression. Any
            detected satellites that fall within this radius will be evaluated
            for suppression, and all of the detections except for the one with
            the highest classification accuracy will be suppressed
        :return: nothing
        """
        #TODO write radial-based non-max suppression

        return NotImplementedError

    def cluster_non_max_suppression(self, locations, scores, thresh=0.9):
        """Custom non-max suppression based on hierarchical clustering.

        :param locations: list of location clusters around each detection
        :param scores: list of object scores around each detection
        :param thresh: threshold on the confidence level required to count
            the detection as a TP
        :return: final object centroids, bounding boxes, and scores
        """
        max_scores = []
        obj_centroids = []
        obj_boxes = []

        x_delta = 0.5 * self.box_size / self.sliding_window.img_width
        y_delta = 0.5 * self.box_size / self.sliding_window.img_height
        if locations:
            for i, location in enumerate(locations):
                ind = np.argmax(scores[i])
                if location.shape[0] == 1:
                    score = float(scores[i])
                else:
                    score = float(scores[i][ind])

                if score >= thresh:
                    max_scores.append(score)
                    obj_centroids.append(location[ind])
                    obj_boxes.append(
                        [(location[ind][0] - y_delta),
                         (location[ind][1] - x_delta),
                         (location[ind][0] + y_delta),
                         (location[ind][1] + x_delta)])

        return obj_centroids, obj_boxes, max_scores

    def plot_final_preds(self, locations, plot_gt=False, plot_centroids=False):
        """Plot the detected objects after non-max suppression.

        :param locations: predicted centroids of the object
        :param scores: predicted class scores of the object
        :param plot_gt: whether to plot the ground truth boxes
        :param plot_centroids: whether to plot the ground truth centroids
        :return: nothing
        """
        x_delta = 0.5 * self.box_size
        y_delta = 0.5 * self.box_size
        plt.imshow(self.sliding_window.image, cmap='gray')
        if any(locations):
            for i, location in enumerate(locations):
                x_min = location[1] * self.sliding_window.img_width - x_delta
                y_min = location[0] * self.sliding_window.img_height - y_delta
                plt.gca().add_patch(Rectangle((x_min, y_min),
                                              self.box_size, self.box_size,
                                              linewidth=1,
                                              edgecolor='r',
                                              facecolor='none'))

            if plot_centroids:
                x_c = []
                y_c = []
                for location in locations:
                    x_c.append(location[1] * self.sliding_window.img_width)
                    y_c.append(location[0] * self.sliding_window.img_height)

                plt.scatter(x=x_c, y=y_c, c='r', s=10)

        else:
            pass

        if plot_gt:
            if self.gt_annos is None:
                pass
            else:
                if any(self.gt_annos.objects):
                    for obj in self.gt_annos.objects:
                        x_min = obj.x_min * self.sliding_window.img_width
                        y_min = obj.y_min * self.sliding_window.img_height
                        plt.gca().add_patch(Rectangle((x_min, y_min),
                                                      self.box_size, self.box_size,
                                                      linewidth=1,
                                                      edgecolor='y',
                                                      facecolor='none'))

                    if plot_centroids:
                        for obj in self.gt_annos.objects:
                            plt.scatter(x=[obj.x_c * self.sliding_window.img_width],
                                        y=[obj.y_c * self.sliding_window.img_height],
                                        c='y', s=10)

        plt.show()

        return

    def non_max_suppression(self, boxes, scores, conf_thresh=0.7, iou_thresh=0.5, max_boxes=10):
        """Traditional non-max suppression.

        :param boxes: the predicted object bounding boxes
        :param scores: the predicted object class scores
        :param conf_thresh: threshold on the scores
        :param iou_thresh: threshold on the IoU overlap between the ground
            truth and predicted boxes
        :param max_boxes: maximum number of boxes that can be predicted
        :return: indices of maximums
        """
        inds = tf.image.non_max_suppression(boxes=boxes, scores=scores,
                                            max_output_size=max_boxes,
                                            iou_threshold=iou_thresh,
                                            score_threshold=conf_thresh)

        return inds


def test():
    img_path = r'C:\Users\jsanders\Desktop\data\seednet2satnet\SatNet\data\rme02.04.22.2015\ImageFiles\sat_36516.0116.fits'
    anno_path = r'C:\Users\jsanders\Desktop\data\seednet2satnet\SatNet\data\rme02.04.22.2015\Annotations\sat_36516.0116.json'
    classifier_path = 'C:/Users/jsanders/Desktop/dlae_migration2/dlae/models/seedNet2satNet_classifier_32w_4s_10p_10r.h5'
    localizer_path = 'C:/Users/jsanders/Desktop/dlae_migration2/dlae/models/seedNet2satNet_localizer_32w_4s_10p_10r.h5'
    image = SatelliteImage(img_path)
    anno = ImageAnnotations(anno_path)

    # pull all object centroids in the image and store in a list
    gt_centroids = []
    gt_boxes = []
    [gt_centroids.append([obj.y_c, obj.x_c]) for obj in anno.objects]
    [gt_boxes.append([obj.y_min, obj.x_min, obj.y_max, obj.x_max]) for obj in anno.objects]

    # run sliding window algorithm across the image
    sw = SatNetSubWindows(img=image.image,
                          centroids=gt_centroids,
                          window_size=32,
                          stride=1,
                          padding=12,
                          img_width=512,
                          img_height=512,
                          pad_img=True)
    sw.z2o_normalize_windows(0.0, 65535.0)
    classifier = load_model(classifier_path)
    localizer = load_model(localizer_path)

    inference_obj = SeedNet2SatNetInference(classifier, localizer, sw, gt_annos=anno)
    inference_obj.plot_raw_inferences(plot_gt=True)
    inference_obj.plot_raw_boxes(plot_gt=True)
    # cluster_locs, cluster_scores = inference_obj.cluster_raw_detections(thresh=0.05)
    # object_locs, object_boxes, object_scores = inference_obj.cluster_non_max_suppression(cluster_locs, cluster_scores)
    # inference_obj.plot_final_preds(object_locs, plot_gt=True, plot_centroids=True)

    inds = inference_obj.non_max_suppression(inference_obj.raw_global_location_boxes,
                                             inference_obj.raw_pred_object_scores,
                                             conf_thresh=0.0, iou_thresh=1.0, max_boxes=10)
    with tf.Session() as sess:
        sess.run(inds)
        detection_inds = inds.eval()

    object_locs = list(inference_obj.raw_global_location_preds[detection_inds])
    object_preds = list(inference_obj.raw_global_location_boxes[detection_inds])
    object_scores = list(inference_obj.raw_pred_object_scores[detection_inds])

    inference_obj.plot_final_preds(object_locs, plot_gt=True, plot_centroids=True)

    # # create the file name
    # file_name = '_'.join([anno.directory, anno.name])
    #
    # # add detections to the dictionary
    # detections_dict['image_name'].append(file_name)
    # detections_dict['predicted_boxes'].append(inference_obj.raw_global_location_boxes)
    # detections_dict['predicted_scores'].append(inference_obj.raw_class_preds)
    # detections_dict['ground_truth_boxes'].append(gt_boxes)
    # detections_dict['ground_truth_class_id'].append(inference_obj.satellite_class_preds)
    #
    # with open(FLAGS.json_path, 'w') as f:
    #     json.dump(detections_dict, f, indent=1)
    #
    # f.close()


if __name__ == '__main__':
    test()
