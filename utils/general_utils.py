

'''seedNet2satNet/utils/general_utils.py

General purpose utility functions.
'''


import json
import h5py
from keras.models import load_model


def load_json(json_file):
    """Reads a JSON file.

    :param json_file: path to the JSON file
    :return: contents of the JSON file
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    f.close()

    return data


def read_hdf5(hdf5_file):
    """Read am HDF5 file with a single data element.

    :param hdf5_file: path to the HDF5 file
    :return: the data element saved in the HDF5 file
    """
    f = h5py.File(hdf5_file, 'r')
    key = list(f.keys())[0]
    data = f[key].value
    f.close()

    return data


def write_hdf5(file_path, dataset):
    """Write a single dataset to an HDF5 file.

    :param file_path: path to save the file
    :param dataset: data to be saved
    :return: nothing
    """
    f = h5py.File(file_path, 'w')
    f.create_dataset(name='data', data=dataset)
    f.close()

    return


def txt2list(file_path):
    """Reads lines from a .txt and store them in a list.

    The .txt file is assumed to contain a series of lines of information.

    :param file_path: path to the .txt to be read
    :return: lines of the .txt file as a list
    """
    file = open(file_path)
    file_lines = file.readlines()
    file_lines = [x.strip() for x in file_lines]

    return file_lines


def ckpt2model(ckpt_path, model_path, return_model=False):
    """Convert a Keras ckeckpoint to a Keras model.

    :param ckpt_path: path to the checkpoint
    :param model_path: path to store the model
    :return: nothing
    """
    model = load_model(ckpt_path)
    model.save(model_path)

    if return_model:
        model = load_model(model_path)
        return model
    else:
        return
