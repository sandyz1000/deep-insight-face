import typing
import logging
import tensorflow as tf
from PIL import Image
from skimage import io
import numpy as np
from . import cfg
from . import yolov3 as yolo
from .yolov3 import YoloArgs
from abc import ABCMeta, abstractmethod
from six import add_metaclass

logger = logging.getLogger(__name__)


class AlignmentInfo(object):
    """
    Get the scaled/cropped images and bounding box co-ordinates
    """

    def __init__(self):
        self.bb = []
        self.scaled_images = []
        self.nrof_successfully_aligned = 0
        self.text_file = []


def to_rgb(img: np.ndarray):
    """ Numpy method to convert GRAYSCALE to RGB
    """
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def filter_bounding_box(img: Image, bounding_boxes: typing.List[typing.List[int]], margin: int = 8):
    """
    # POST PROCESSING FILTER BOUNDING BOX
    """
    assert isinstance(img, Image.Image), "Invalid image type"
    img = np.array(img)  # cvt to numpy array
    det = np.array(bounding_boxes)
    det_arr = []
    img_size = np.asarray(img.shape)[0:2]
    det_arr = [np.squeeze(det[i]) for i in range(len(bounding_boxes))]
    cropped_images, boxes = [], []
    for i, det in enumerate(det_arr):
        det = np.squeeze(det)
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
        cropped_img = img[bb[1]:bb[3], bb[0]:bb[2], :]
        # cropped_img = Image.fromarray(cropped_img).resize((image_size, image_size), resample=Image.BILINEAR)
        scaled = np.asarray(cropped_img)
        boxes.append(bb)
        cropped_images.append(scaled)
    return cropped_images, boxes


def get_bounding_box(infer_model, img_path: str, anchors: typing.List, num_classes: int):
    """
    Get the bounding box for the given image
    :param infer_model: [description]
    :type infer_model: [type]
    :param img_path: [description]
    :type img_path: [type]
    :param anchors: [description]
    :type anchors: [type]
    :param num_classes: [description]
    :type num_classes: [type]
    :return: [description]
    :rtype: [type]
    """    
    image = Image.open(img_path).convert('RGB')
    boxed_image = yolo.letterbox_image(image, (416, 416))
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)
    yolo_output = infer_model.predict(image_data)
    yolo_output = [tf.convert_to_tensor(out, dtype=tf.float32) for out in yolo_output]
    image_shape = (image.size[1], image.size[0])
    boxes, scores, classes = yolo.evaluate(yolo_output, anchors, num_classes, image_shape)
    # TODO: Get boxes with highest scores
    # final_boxes = detect_boxes(boxes, scores, classes, image_shape)
    final_boxes = [(left, top, right, bottom) for top, left, bottom, right in boxes]
    return final_boxes, image


@add_metaclass(ABCMeta)
class _BaseDetection:

    @abstractmethod
    def detect_bbox(self, img: typing.Union[str, np.ndarray]):
        """
        Align given image and return a scaled image to be used for farther encoding.

        A base method that need to be implementated by both YOLO and MTCNN Detector

        :param img: A numpy array or string contains actual path of image
        :raises ValueError: If faces not found raise Error
        :return: Return Crop image and bounding box
        """
        raise NotImplementedError


class YoloDetection(_BaseDetection):

    def __init__(self, margin: int = 8,
                 detect_multiple_faces: bool = False, image_size: int = 416, **kwargs) -> None:
        self.margin = margin
        self.detect_multiple_faces = detect_multiple_faces
        self.image_size = image_size
        self.model_path = kwargs['model_path']
        self.score = kwargs.pop("score", 0.4)
        self.iou = kwargs.pop("iou", 0.5)
        self.anchors = kwargs.pop("anchors", cfg.YOLO_ANCHORS)
        self.anchors = np.array(self.anchors, dtype=int)
        self.classes = kwargs.pop("classes", cfg.YOLO_CLASSES)
        # self.kwargs = kwargs
        args = YoloArgs(yolo_model=self.model_path, anchors=self.anchors, classes=self.classes,
                        img_size=(self.image_size, self.image_size))
        self.__init_detector__(args)

    def __init_detector__(self, args):
        print('Creating networks and loading parameters')
        self.myYolo = yolo.YOLO(args)

    def __call__(self, img: typing.Union[str, np.ndarray]):
        return self.detect_bbox(img)

    def detect_bbox(self, img: typing.Union[str, np.ndarray]):
        if isinstance(img, str):
            img = io.imread(img)
        if img.ndim < 2:
            message = f'Unable to align {img.shape}'
            raise ValueError(message)

        if img.ndim == 2:
            img = to_rgb(img)
        img = img[:, :, 0:3]

        image = Image.fromarray(img)  # Convert to PIL Image
        _, bounding_boxes = self.myYolo.detect_boxes(image)

        logger.info("Found bounding box, ", str(bounding_boxes))
        nrof_faces = len(bounding_boxes)
        if nrof_faces > 0:
            cropped = filter_bounding_box(img, nrof_faces, bounding_boxes,
                                          do_save=False,
                                          detect_multiple_faces=self.detect_multiple_faces)
            # bounding_boxes = [np.array([bb[1], bb[2], bb[3], bb[0]]) for bb in cropped.bb]
            return cropped.scaled_images, cropped.bb

        raise ValueError("Bounding box not found")
