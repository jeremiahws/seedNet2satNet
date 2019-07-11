

'''seedNet2satNet/utils/patch_extractor.py

Class to extract and prepare sub-windows from SatNet images.
'''


import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing
import itertools


def extract_patch(img, corner_coords):
    patch = img[corner_coords[0]:corner_coords[1], corner_coords[2]:corner_coords[3]]

    return patch


class SatNetSubWindows(object):
    """Performs sliding window algorithm across a satellite image.

        Attributes:
            img (int): the image from which the sub-windows are extracted from
            window_size (int): size of the sub-windows (in pixels)
            img_width (int): number of columns of the image. If not specified,
                it is assumed to be 512
            img_height (int): number of rows of the image. If not specified,
                it is assumed to be 512
            img_pad (int): amount of padding applied to the image (for edge
                cases)
            n_windows (int): number of sub-windows extracted from the image
            windows (int): sub-windows extracted from the image, stored as a
                uint16 numpy array
            window_corner_coords (int): coordinates of the top left pixel in
                the sub-window relative to the top left pixel of the entire
                image
            object_present (int): whether or not an object is present in the
                sub-window (0 for no objects, 1 for object(s) present). This
                vector can be used as the class labels of the sub-windows for
                training a classifier
            object_location (float): object location within the sub-window
                relative to the top left corner of the sub-window. These
                labels can be used for training a localization model. Adding
                the object_location and the window_corner_coords will yield
                the object's global image coordinates with respect to the top
                left pixel of the image
            windows_with (float or None): sub-windows with an obejct present
                in the window. To populate this variable, a call must be made
                to the get_obj_windows method
            windows_without (float or None): sub-windows without an obejct
                present in the window. To populate this variable, a call must
                be made to the get_obj_windows method
    """
    def __init__(self, img, centroids, window_size, stride, padding=0, img_width=512, img_height=512,
                 pad_img=True, parallel=True):
        """Performs sliding-window upon initialization.

        :param img: the satellite image
        :param centroids: centroids of the objects in the image
        :param window_size: size of the sub-windows (in pixels)
        :param stride: stride of the sliding window (in pixels). Setting
            stride < window_size will yield overlapping sub-windows
        :param padding: padding to apply to the windows to restrict the
            classification of the window. If padding > 0, the window will be
            classified as a positive window only if the object falls within
            the central potion of the window, outside of the padded region
        :param img_width: number of columns of the image
        :param img_height: number of rows of the image
        :param pad_img: whether or not to add padding to the image before
            performing the sliding window operation on the image. Useful for
            cases where there may be an object near the edge(s) of the image
        :param parallel: whether to perform the sub-window extraction in
            parallel
        """
        self.image = img
        self.window_size = window_size
        self.img_width = img_width
        self.img_height = img_height
        self.stride = stride

        if pad_img:
            self.img_pad = padding
            img = np.pad(self.image, pad_width=self.img_pad, mode='constant')
        else:
            self.img_pad = 0

        n_row_windows = np.ceil((self.img_height + 2 * self.img_pad - self.window_size + self.stride) / self.stride).astype(int)
        n_col_windows = np.ceil((self.img_width + 2 * self.img_pad - self.window_size + self.stride) / self.stride).astype(int)
        n_windows = n_row_windows * n_col_windows

        self.n_windows = n_windows
        self.windows = np.zeros([n_windows, self.window_size, self.window_size], dtype='uint16')
        self.window_corner_coords = np.empty([n_windows, 2], dtype='float32')
        self.object_present = np.zeros(n_windows, dtype='uint8')
        self.object_location = np.empty([n_windows, 2], dtype='float32')
        self.windows_with = None
        self.windows_without = None
        self.object_location_with = None
        self.object_location_without = None

        if parallel:
            rstart = [i for i in range(0, n_row_windows * self.stride, self.stride)]
            rend = [i + self.window_size for i in rstart]
            cstart = [i for i in range(0, n_col_windows * self.stride, self.stride)]
            cend = [i + self.window_size for i in rstart]
            if rend[-1] > self.img_height + 2 * self.img_pad - 1:
                rstart[-1] = self.img_height + 2 * self.img_pad - self.window_size
                rend[-1] = None

            if cend[-1] > self.img_width + 2 * self.img_pad - self.window_size:
                cstart[-1] = self.img_width + 2 * self.img_pad - self.window_size
                cend[-1] = None

            row_coords = np.transpose(np.array([rstart, rend]))
            col_coords = np.transpose(np.array([cstart, cend]))

            corner_coords = []
            for (r, c) in itertools.product(*(row_coords, col_coords)):
                corner_coords.append(np.concatenate((r, c)))

            num_cores = multiprocessing.cpu_count()
            windows = Parallel(n_jobs=num_cores)(delayed(extract_patch)(img, coord) for coord in corner_coords)
            self.windows = np.asarray(windows, dtype=np.uint16)
            self.window_corner_coords = np.asarray(corner_coords)

            #TODO get centroids
            # for centroid in centroids:
            #     if (rwindow_start + padding) / self.img_height < centroid[0] < (rwindow_start + self.window_size - padding) / self.img_height \
            #             and (cwindow_start + padding) / self.img_width < centroid[1] < (cwindow_start + self.window_size - padding) / self.img_width:
            #         self.object_present[count] = 1
            #         self.object_location[count, :] = [centroid[0] - rwindow_start / self.img_height,
            #                                           centroid[1] - cwindow_start / self.img_width]
            #     else:
            #         self.object_present[count] = 0
            #         self.object_location[count, :] = [0.0, 0.0]

        else:
            count = 0
            for cwindow in range(n_col_windows):
                if cwindow == 0:
                    cwindow_start = cwindow
                    cwindow_end = self.window_size
                else:
                    cwindow_start = cwindow_start + stride
                    cwindow_end = cwindow_start + self.window_size

                if cwindow_end > self.img_width + 2 * self.img_pad - 1:
                    cwindow_start = self.img_width + 2 * self.img_pad - self.window_size
                    cwindow_end = None

                for rwindow in range(n_row_windows):
                    if rwindow == 0:
                        rwindow_start = rwindow
                        rwindow_end = self.window_size
                    else:
                        rwindow_start = rwindow_start + stride
                        rwindow_end = rwindow_start + self.window_size

                    if rwindow_end > self.img_height + 2 * self.img_pad - 1:
                        rwindow_start = self.img_height + 2 * self.img_pad - self.window_size
                        rwindow_end = None

                    self.windows[count, :, :] = img[rwindow_start:rwindow_end, cwindow_start:cwindow_end]
                    self.window_corner_coords[count, :] = [(rwindow_start - self.img_pad) / self.img_height,
                                                           (cwindow_start - self.img_pad) / self.img_width]

                    for centroid in centroids:
                        if (rwindow_start + padding) / self.img_height < centroid[0] < (rwindow_start + self.window_size - padding) / self.img_height \
                                and (cwindow_start + padding) / self.img_width < centroid[1] < (cwindow_start + self.window_size - padding) / self.img_width:
                            self.object_present[count] = 1
                            self.object_location[count, :] = [centroid[0] - rwindow_start / self.img_height, centroid[1] - cwindow_start / self.img_width]
                        else:
                            self.object_present[count] = 0
                            self.object_location[count, :] = [0.0, 0.0]
                    count = count + 1

        self.windows = self.windows[:, :, :, np.newaxis]

    def get_obj_windows(self):
        """Separate windows with/without objects present.

        :return: nothing
        """
        bg_inds = np.where(self.object_present == 0)
        sat_inds = np.where(self.object_present == 1)
        self.windows_without = np.squeeze(self.windows[bg_inds, :, :])
        self.object_location_without = np.squeeze(self.object_location[bg_inds, :])
        if sat_inds[0].size > 0:
            self.windows_with = np.squeeze(self.windows[sat_inds, :, :])
            self.object_location_with = np.squeeze(self.object_location[sat_inds, :])

        if sat_inds[0].size == 1:
            self.windows_with = self.windows_with[np.newaxis, :, :]
            self.object_location_with = self.object_location_with[np.newaxis, :]

        return

    def plot_one_with(self):
        """Plot a random window with an object present.

        :return: nothing
        """
        try:
            ind = np.random.randint(self.windows_with.shape[-1])
            plt.imshow(self.windows_with[ind, :, :])
            plt.scatter(x=[self.object_location_with[ind, 1] * self.img_width],
                        y=[self.object_location_with[ind, 0] * self.img_height],
                        c='r', s=30)
            plt.show()
        except ValueError:
            print('No windows to show - try running the get_obj_windows method.')

        return

    def plot_one_without(self):
        """Plot a random window without an object present.

        :return: nothing
        """
        try:
            ind = np.random.randint(self.windows_without.shape[-1])
            plt.imshow(self.windows_without[ind, :, :])
            plt.show()
        except ValueError:
            print('No windows to show - try running the get_obj_windows method.')

        return

    def z2o_normalize_windows(self, min, max):
        """Normalize the sub-windows to fall in the range [0, 1]. The specific
        normalization applied is new = (old - min) / (max - min).

        :param min: the minimum intensity value
        :param max: the maximum intensity value
        :return: nothing
        """
        self.windows = (self.windows - min) / (max - min)
        if self.windows_with is not None:
            self.windows_with = (self.windows_with - min) / (max - min)
        if self.windows_without is not None:
            self.windows_without = (self.windows_without - min) / (max - min)
