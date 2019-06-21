

'''seedNet2satNet/utils/json_parser.py

Reads a JSON file containing object information.
'''


from utils.general_utils import load_json


class ObjectFeatures(object):
    """Defining features of the objects contained within a .fits file.

        Attributes:
            class_name (str): name of the object class
            class_id (int): ID of the object class
            x_min (float): coordinate of the bounding box closest to the top
                left pixel of the image defined in the left-right direction.
                The value is defined as a fraction of the total number of
                image columns and is measured in reference to the top left
                pixel of the image
            y_min (float): coordinate of the bounding box closest to the top
                left pixel of the image defined in the top-bottom direction.
                The value is defined as a fraction of the total number of
                image rows and is measured in reference to the top left
                pixel of the image
            x_max (float): coordinate of the bounding box furthest from the
                top left pixel of the image defined in the left-right
                direction. The value is defined as a fraction of the total
                number of image columns and is measured in reference to the
                top left pixel of the image
            y_max (float): coordinate of the bounding box furthest from the
                top left pixel of the image defined in the top-bottom
                direction. The value is defined as a fraction of the total
                number of image rows and is measured in reference to the top
                left pixel of the image
            x_c (float): coordinate of the object centroid defined in the
                left-right direction. The value is defined as a fraction of
                the total number of image columns and is measured in reference
                to the top left pixel of the image
            y_c (float): coordinate of the object centroid defined in the
                top-bottom direction. The value is defined as a fraction of
                the total number of image rows and is measured in reference to
                the top left pixel of the image
            bbox_width (float): width of the bounding box around the object,
                defined as a fraction of the total number of image columns
            bbox_height (float): height of the bounding box around the object,
                defined as a fraction of the total number of image rows
            source (str): source of the detection
            magnitude (float): magnitude of the detection
    """
    def __init__(self, obj):
        """Extracts features from an object structure upon initialization.

        :param obj: object structure from JSON annotations file
        """
        self.class_name = obj['class_name']
        self.class_id = obj['class_id']
        self.x_min = obj['x_min']
        self.y_min = obj['y_min']
        self.x_max = obj['x_max']
        self.y_max = obj['y_max']
        self.x_c = obj['x_center']
        self.y_c = obj['y_center']
        self.bbox_width = obj['bbox_width']
        self.bbox_height = obj['bbox_height']
        self.source = obj['source']
        self.magnitude = obj['magnitude']


class ImageAnnotations(object):
    """Information from JSON file of objects contained within a .fits file.

        Attributes:
            directory (str): directory of sensor data where image is stored
            name (str): file name for the .fits image within the directory
            sequence (list): list of the 6 image names that comprise the
                "collect"
            sequence_id (int): ID of the image sequence
            height (int): height of the image (in pixels)
            width (int): width of the image (in pixels)
            fov_x (float): x-dimension of the image FOV
            fov_y (float): y-dimension of the image FOV
            objects (list): list of the objects within the image. Each object
                contains a set of features defining the object including its
                location in the image and the bounding box around the object
    """
    def __init__(self, json_path):
        """Reads and sorts JSON annotations file upon initialization.

        :param json_path: path to the JSON annotations file
        """
        contents = load_json(json_path)
        self.directory = contents['data']['file']['dirname']
        self.name = contents['data']['file']['filename']
        self.sequence = contents['data']['file']['sequence']
        self.sequence_id = contents['data']['file']['sequence_id']
        self.height = contents['data']['sensor']['height']
        self.width = contents['data']['sensor']['width']
        self.fov_x = contents['data']['sensor']['iFOVx']
        self.fov_y = contents['data']['sensor']['iFOVy']
        self.objects = []
        [self.objects.append(ObjectFeatures(obj)) for obj in contents['data']['objects']]
