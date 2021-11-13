import typing
import tensorflow as tf
from keras import backend as K
import cv2
import numpy as np
import colorsys
# from .utility import (preprocess_input, decode_netout, draw_boxes,
#                       correct_yolo_boxes, do_nms, thread_safe_singleton)
from threading import Lock
from PIL import Image
import warnings
warnings.simplefilter('ignore')


def thread_safe_singleton(cls):
    instances = {}
    session_lock = Lock()

    def wrapper(*args, **kwargs):
        with session_lock:
            if cls not in instances:
                instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return wrapper


class YoloArgs(typing.NamedTuple):
    model: typing.Any
    anchors: np.ndarray
    classes: int
    obj_thresh: float = 0.3
    nms_thresh: float = 0.45
    img_size: typing.Tuple = (416, 416)


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    '''Convert final layer features to bounding box parameters'''

    num_anchors = len(anchors)

    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    # height, width
    grid_shape = K.shape(feats)[1:3]
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                    [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                    [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1],
                                                         K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1],
                                                              K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    if calc_loss:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''Get corrected boxes'''

    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


def boxes_and_scores(feats, anchors, num_classes, input_shape,
                     image_shape):
    '''Process Convolutional layer output'''

    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats, anchors, num_classes, input_shape)
    boxes = correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def letterbox_image(image: Image, size: typing.Tuple[int]) -> Image:
    '''Resize image with unchanged aspect ratio using padding'''
    img_width, img_height = image.size
    w, h = size
    scale = min(w / img_width, h / img_height)
    nw = int(img_width * scale)
    nh = int(img_height * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


def get_yolo_output(outputs: typing.List[tf.Tensor],
                    anchors: np.ndarray,
                    num_classes: int,
                    image_shape: typing.Tuple,
                    max_boxes: int = 20,
                    score_threshold: float = .6,
                    iou_threshold: float = .5):
    '''Evaluate the YOLO model on given input and return filtered boxes'''

    num_layers = len(outputs)
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [
        [3, 4, 5], [1, 2, 3]]
    input_shape = K.shape(outputs[0])[1:3] * 32
    boxes = []
    box_scores = []

    for l in range(num_layers):
        _boxes, _box_scores = boxes_and_scores(outputs[l],
                                               anchors[anchor_mask[l]],
                                               num_classes, input_shape,
                                               image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)

    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []

    for c in range(num_classes):
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor,
            iou_threshold=iou_threshold)
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)

    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_


def draw(frame, left, top, right, bottom):
    cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), 1)


def get_masked_image(frame, classwise_tuples):
    frame = np.asarray(frame)
    keys = classwise_tuples.keys()
    for key in keys:
        draw(frame, key[0], key[1], key[0] + key[2], key[1] + key[3])
    return frame


def preprocess_input(image: np.ndarray, net_h: int, net_w: int):
    new_h, new_w, _ = image.shape

    # determine the new size of the image
    if (float(net_w) / new_w) < (float(net_h) / new_h):
        new_h = (new_h * net_w) // new_w
        new_w = net_w
    else:
        new_w = (new_w * net_h) // new_h
        new_h = net_h

    # resize the image to the new size
    # resized = cv2.resize(image[:, :, ::-1] / 255., (new_w, new_h))
    resized = cv2.resize(image / 255., (new_w, new_h))

    # embed the image into the standard letter box
    new_image = np.ones((net_h, net_w, 3)) * 0.5
    new_image[(net_h - new_h) // 2:(net_h + new_h) // 2, (net_w - new_w) // 2:(net_w + new_w) // 2, :] = resized
    new_image = np.expand_dims(new_image, 0)

    return new_image
