

'''seedNet2satNet/utils/image_reader.py

Class to create an image object from a .fits file.
'''


from astropy.io import fits
import numpy as np
import os


class SatelliteImage(object):
    """.fits image and header information.

        Attributes:
            image (float): the satellite image stored as a float32 numpy array
            header (list): header information of the .fits file
            name (str): name of the .fits file
    """
    def __init__(self, file_path):
        """Reads .fits file upon initialization.

        :param file_path: path to the .fits image
        """
        file_parts = os.path.split(file_path)
        # the file will be the last element
        self.name = file_parts[-1]

        hdulist = fits.open(file_path)
        hdu = hdulist[0]
        self.image = np.asarray(hdu.data, 'float32')
        self.header = hdu.header
