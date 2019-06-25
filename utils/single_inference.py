

'''seedNet2satNet/utils/single_inference.py

Performs inference on a SatNet image.
'''


import numpy as np
from operator import add
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import scipy.cluster.hierarchy as hcluster
import tensorflow as tf


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
            tp_windows (float): true positive sub-windows
            tn_windows (float): true negative sub-windows
            fp_windows (float): false positive sub-windows
            fn_windows (float): false negative sub-windows
    """
    def __init__(self,
                 classification_model,
                 localization_model,
                 sliding_window,
                 batch_size=64,
                 box_size=20,
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
        self.box_size = box_size # int
        self.gt_annos = gt_annos # image anno object
        self.sliding_window = sliding_window # sliding window object
        self.raw_class_preds = classification_model.predict(sliding_window.windows, batch_size=batch_size) # 2D numpy array
        self.class_preds = np.argmax(self.raw_class_preds, axis=1) # 1D numpy array

        inds = np.where(self.class_preds == 1) # tuple with first entry as numpy array
        self.n_detections = len(inds[0]) # int
        self.satellite_class_preds = self.class_preds[inds].astype(float) # 1D numpy array
        self.raw_pred_object_scores = np.squeeze(self.raw_class_preds[inds, 1]) # 1D numpy array
        self.satellite_window_preds = sliding_window.windows[inds] # 4D numpy array, last dim = 1 (1 channel)
        self.satellite_window_gt = sliding_window.object_present[inds] # 1D numpy array
        self.satellite_window_gt_location = sliding_window.object_location[inds] # 2D numpy array
        self.satellite_window_corner_coords = sliding_window.window_corner_coords[inds] # 2D numpy array

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

        else:
            pass

    def plot_raw_inferences(self, plot_gt=False):
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
        if locations:
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
