import os
from easydict import EasyDict
from .networks.utils import set_gpu_limit

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
__C = EasyDict()
cfg = __C
__C.BASE_DIR = BASE_DIR

WEIGHTS_DIR = os.makedirs(os.path.expanduser("~/deep_insight_weights"), exist_ok=True)
# config = '/data/darknet/cfg/yolo-single_class.cfg'
# classes = '/data/darknet/data/single.names'
# WEIGHTS_DIR = os.path.join(BASE_DIR, 'weights')


# YOLO config for face-detection
CROPPED_PATH = "cropped-output"
YOLO_MODEL = os.path.join(WEIGHTS_DIR, "detector/YOLO_Face.h5")
YOLO_ANCHORS = os.path.join(os.path.dirname(__file__), "yolo_cfg/yolo_anchors.txt")
YOLO_CLASSES = os.path.join(os.path.dirname(__file__), "yolo_cfg/face_classes.txt")

GPU_SIZE_LIMIT = 2  # IN GB
set_gpu_limit(GPU_SIZE_LIMIT)


__DATA_DIR = args['DATA_DIR'] if 'DATA_DIR' in args else BASE_DIR
__choice = EasyDict(bottleneck="", path="", image_size=(), emd_size=128)

__C.FACERECO_CFG = EasyDict()
__C.WEIGHTS_DIRNAME = args['WEIGHTS_DIRNAME']
__C.FACERECO_CFG.BOTTLENECK = __choice.bottleneck
__C.FACERECO_CFG.IMG_SIZE = __choice.image_size
__C.FACERECO_CFG.EMD_SIZE = __choice.emd_size
__C.FACERECO_CFG.MODULE = args['FACERECO_MODULE'].upper()
__C.FACERECO_CFG.FACEMATCH_PATH = os.path.join(__DATA_DIR, __C.WEIGHTS_DIRNAME, __choice.path)
__C.LANDMARK_PATH = os.path.join(__DATA_DIR, __C.WEIGHTS_DIRNAME, args['LANDMARK_MODEL_PATH'])
__C.DETECTOR_MODEL = args['DETECTOR_MODEL']  # YoLov3 and MTCNN
__C.YOLO_PATH = os.path.realpath(__DATA_DIR + os.path.sep + __C.WEIGHTS_DIRNAME +
                                 os.sep + args['DETECTOR_MODEL_PATH'])
