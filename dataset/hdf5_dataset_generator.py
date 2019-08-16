

"""seedNet2satNet/hdf5_dataset_generator.py

Dataset generator for HDF5 files of sub-windows.
"""


import h5py
import numpy as np
from random import shuffle


def get_keys(path):
    f = h5py.File(path, 'r')
    f_keys = list(f.keys())
    f.close()

    return f_keys


class DatasetGenerator(object):
    def __init__(self,
                 hdf5_path,
                 parse_function,
                 augment=False,
                 shuffle=False,
                 batch_size=4,
                 negpos_ratio=1):
        """
        Constructor for the data generator class. Takes as inputs many configuration choices, and returns a generator
        with those options set.
        :param hdf5_path: the name of the HDF5 to be processed.
        :param parse_function: tfrecords parsing function.
        :param augment: whether or not to apply augmentation to the processing chain.
        :param shuffle: whether or not to shuffle the input buffer.
        :param batch_size: the number of examples in each batch produced.
        """
        self.hdf5_path = hdf5_path
        self.batch_size = batch_size
        self.parse_function = parse_function
        self.keys = get_keys(self.hdf5_path)
        self.shuffle = shuffle
        self.augment = augment
        self.z2o_normalization = 2**16
        self.negpos_ratio = negpos_ratio

        self.file_names = []
        for key in self.keys:
            splits = key.split(':')
            self.file_names.append(splits[0])

        self.file_names = list(set(self.file_names))
        self.dataset_size = len(self.file_names)
        self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32)

    def __len__(self):
        """
        The "length" of the generator is the number of batches expected.
        :return: the expected number of batches that will be produced by this generator.
        """
        return int(np.ceil(self.dataset_size / self.batch_size))

    def generate(self):
        """
        Reads in data from an HDF5 file, applies augmentation chain (if
        desired), shuffles and batches the data.
        """
        if self.shuffle:
            shuffle(self.file_names)

        current = 0

        while True:
            batch_X, batch_y = [], []

            if current >= self.dataset_size:
                current = 0

                if self.shuffle:
                    shuffle(self.file_names)

            batch_indices = self.dataset_indices[current:current + self.batch_size]
            f = h5py.File(self.hdf5_path, 'r')
            for i in batch_indices:
                windows_name = self.file_names[i] + ":windows"
                annos_name = self.file_names[i] + ":annos"
                windows = f[windows_name].value
                annos = f[annos_name].value
                for window in windows:
                    batch_X.append(window / self.z2o_normalization)

                for anno in annos:
                    batch_y.append(anno)

                if self.augment:
                    #TODO augmentation functions
                    pass
            f.close()

            batch_X = np.array(batch_X)
            batch_X = np.expand_dims(batch_X, axis=-1)
            batch_y = np.array(batch_y)
            sample_weights = (batch_y * (self.negpos_ratio - 1)) + 1

            current += self.batch_size

            ret = (batch_X, batch_y, sample_weights)

            yield ret
