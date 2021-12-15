import os
import typing
import logging
import tensorflow as tf
from PIL import Image
from skimage import io
import numpy as np
from . import yolov3 as yolo
from .yolov3 import YoloArgs

logger = logging.getLogger(__name__)


def _get_class():
    classes_path = os.path.expanduser(os.path.realpath(os.path.join(
        os.path.dirname(__file__), "yolo_cfg/face_classes.txt")))
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def _get_anchors():
    anchors_path = os.path.expanduser(os.path.realpath(os.path.join(
        os.path.dirname(__file__), "yolo_cfg/yolo_anchors.txt")))
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = np.array([float(x) for x in anchors.split(',')])
    return anchors
    # return anchors.reshape(-1, 2)


def _to_rgb(img: np.ndarray):
    """ Numpy method to convert GRAYSCALE to RGB
    """
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def _detect_multiple_boxes(det: np.ndarray, detect_multiple_faces: bool = False, ):
    # TODO: Fix this methods
    det_arr = []
    if nrof_faces > 1:
        if detect_multiple_faces:
            # det_arr = [np.squeeze(det[i]) for i in range(nrof_faces)]
            largest = np.argmax([(det[i, 2] - det[i, 0]) * (det[i, 3] - det[i, 1]) for i in range(nrof_faces)])
            det_arr.append(det[largest, :])
        else:
            bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = img_size / 2
            offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                 (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            # some extra weight on the centering
            index = np.argmax(bounding_box_size - offset_dist_squared * 2.0)
            det_arr.append(det[index, :])
    else:
        det_arr.append(np.squeeze(det))
    return det_arr


def filter_bounding_box(img: Image, bounding_boxes: typing.List[typing.List[int]],
                        margin: int = 8, detect_multiple_faces: bool = False):
    """POST PROCESSING FILTER BOUNDING BOX
    """
    assert isinstance(img, Image.Image), "Invalid image type"
    img = np.array(img)  # cvt to numpy array
    det = np.array(bounding_boxes)
    det_arr = []
    img_size = np.asarray(img.shape)[0:2]
    # TODO: _detect_multiple_boxes Fix this method
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


def get_bounding_box(infer_model, image: Image, anchors: typing.List,
                     num_classes: int, target_size: typing.Tuple = (416, 416)):
    """
    Get the bounding box for the given image
    :param infer_model: [description]
    :type infer_model: [type]
    :param image: 3d Numpy array
    :type image: np.ndarray
    :param anchors: [description]
    :type anchors: [type]
    :param num_classes: [description]
    :type num_classes: [type]
    :return: [description]
    :rtype: [type]
    """
    boxed_image = yolo.letterbox_image(image, target_size)
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)
    yolo_output = infer_model.predict(image_data)
    yolo_output = [tf.convert_to_tensor(out, dtype=tf.float32) for out in yolo_output]
    image_shape = (image.size[1], image.size[0])
    boxes, scores, classes = yolo.get_yolo_output(yolo_output, anchors, num_classes, image_shape)
    # TODO: Get boxes with highest scores
    # final_boxes = detect_boxes(boxes, scores, classes, image_shape)
    final_boxes = [(left, top, right, bottom) for top, left, bottom, right in boxes]
    return final_boxes, boxed_image


class YoloDetection:
    """
    Align given image and return a scaled image to be used for farther encoding.

    A base method that need to be implementated by both YOLO and MTCNN Detector

    """

    def __init__(self, margin: int = 8,
                 detect_multiple_faces: bool = False, image_size: int = 416, **kwargs) -> None:

        self.margin = margin
        self.detect_multiple_faces = detect_multiple_faces
        self.image_size = image_size
        self.model_path = kwargs['model_path']
        self.score = kwargs.pop("score", 0.4)
        self.iou = kwargs.pop("iou", 0.5)
        self.anchors = kwargs.pop("anchors", _get_anchors())
        self.anchors = np.array(self.anchors, dtype=int)
        self.classes = kwargs.pop("classes", _get_class())

        self.args = YoloArgs(yolo_model=tf.keras.models.load_model(self.model_path),
                             anchors=self.anchors,
                             classes=self.classes,
                             img_size=(self.image_size, self.image_size))

    def __call__(self, img: np.ndarray):
        """ Return cropped image and bounding box

        :param img: A numpy array or string contains actual path of image
        :raises ValueError: If faces not found raise Error
        :return: Return Crop image and bounding box
        """
        # TODO: Check this method
        assert isinstance(img, np.ndarray), "Invalid image format"
        if img.ndim < 2:
            message = f'Unable to align {img.shape}'
            raise ValueError(message)

        if img.ndim == 2:
            img = _to_rgb(img)
        img = img[:, :, 0:3]

        # image = Image.fromarray(img)  # Convert to PIL Image
        bounding_boxes, boxed_image = get_bounding_box(img)

        logger.info("Found bounding box, ", str(bounding_boxes))
        nrof_faces = len(bounding_boxes)
        if nrof_faces > 0:
            cropped = filter_bounding_box(boxed_image, bounding_boxes, detect_multiple_faces=self.detect_multiple_faces)
            # bounding_boxes = [np.array([bb[1], bb[2], bb[3], bb[0]]) for bb in cropped.bb]
            return cropped.scaled_images, cropped.bb

        raise ValueError("Bounding box not found")
