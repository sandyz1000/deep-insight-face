import cv2
import typing
import logging
from PIL import Image
from skimage import io
import numpy as np
from .utility import filter_bounding_box, to_rgb
from . import cfg
from . import yolov3 as yolo
from .yolov3 import YoloArgs
from abc import ABCMeta, abstractmethod
from six import add_metaclass
from mtcnn import MTCNN

logger = logging.getLogger(__name__)


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
        self.__init_detector(args)

    def __init_detector(self, args):
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


class MtcnnDetection(_BaseDetection):
    def __init__(self, detect_multiple_faces: bool = False, image_size: int = 416, **kwargs) -> None:
        self.detect_multiple_faces = detect_multiple_faces
        self.image_size = image_size
        self.mtcnn = MTCNN()

    def __call__(self, img: typing.Union[str, np.ndarray]):
        return self.detect_bbox(img)

    def detect_bbox(self, img: typing.Union[str, np.ndarray]):
        if isinstance(img, str):
            img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)

        bounding_boxes = self._format_detection(self.mtcnn.detect_faces(img))
        nrof_faces = len(bounding_boxes)
        if nrof_faces > 0:
            cropped = filter_bounding_box(img, nrof_faces, bounding_boxes,
                                          do_save=False,
                                          detect_multiple_faces=self.detect_multiple_faces)
            # bounding_boxes = [np.array([bb[1], bb[2], bb[3], bb[0]]) for bb in cropped.bb]
            return cropped.scaled_images, cropped.bb

        raise ValueError("Bounding box not found")

    def _format_detection(self, mtcnn_output: typing.List[typing.Dict]):
        boxes, keypoints, confidence = zip(*[(detection['box'], detection['keypoints'], detection['confidence'])
                                             for detection in mtcnn_output])
        boxes = [[bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]] for bb in boxes]
        return boxes
