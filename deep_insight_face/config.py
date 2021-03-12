import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# config = '/data/darknet/cfg/yolo-single_class.cfg'
# classes = '/data/darknet/data/single.names'
WEIGHTS_DIR = os.path.join(BASE_DIR, 'weights')


# YOLO config for face-detection
CROPPED_PATH = "cropped-output"
YOLO_MODEL = os.path.join(WEIGHTS_DIR, "detector/YOLO_Face.h5")
YOLO_ANCHORS = os.path.join(os.path.dirname(__file__), "yolo_cfg/yolo_anchors.txt")
YOLO_CLASSES = os.path.join(os.path.dirname(__file__), "yolo_cfg/face_classes.txt")
