import os
from easydict import EasyDict
import numpy as np


__C = EasyDict()
cfg = __C
__C.ROOT_DIR = os.path.realpath(os.path.dirname(__file__) + os.sep + os.pardir + os.sep + os.pardir)

__C.YOLO_MODEL = os.path.realpath(__C.ROOT_DIR + os.path.sep + "weights/detector/YOLO_Face.h5")

__ANCHOR_PATH = os.path.realpath(os.path.dirname(__file__) + os.path.sep + "yolo_cfg/yolo_anchors.txt")
__FACE_CLASSES_PATH = os.path.realpath(os.path.dirname(__file__) + os.path.sep + "yolo_cfg/face_classes.txt")
__DEFAULT_ANCHOR = ["10", "13", "16", "30", "33", "23", "30", "61",
                    "62", "45", "59", "119", "116", "90", "156", "198", "373", "326"]


def _get_class():
    classes_path = os.path.expanduser(__FACE_CLASSES_PATH)
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def _get_anchors():
    anchors_path = os.path.expanduser(__ANCHOR_PATH)
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = np.array([float(x) for x in anchors.split(',')])
    return anchors
    # return anchors.reshape(-1, 2)


__C.YOLO_ANCHORS = _get_anchors()
__C.YOLO_CLASSES = _get_class()
