import os
import cv2
import numpy as np
from PIL import Image
from skimage import io
from threading import Lock
from scipy.special import expit

BBOX_MARGIN = 0.024
COLORS = [
    [31, 0, 255],
    [0, 159, 255],
    [255, 95, 0],
    [255, 19, 0],
    [255, 0, 0],
    [255, 38, 0],
    [0, 255, 25],
    [255, 0, 133],
    [255, 172, 0],
    [108, 0, 255],
    [0, 82, 255],
    [0, 255, 6],
    [255, 0, 152],
    [223, 0, 255],
    [12, 0, 255],
    [0, 255, 178],
    [108, 255, 0],
    [184, 0, 255],
    [255, 0, 76],
    [146, 255, 0],
    [51, 0, 255],
    [0, 197, 255],
    [255, 248, 0],
    [255, 0, 19],
    [255, 0, 38],
    [89, 255, 0],
    [127, 255, 0],
    [255, 153, 0],
    [0, 255, 255],
    [0, 255, 216],
    [0, 255, 121],
    [255, 0, 248],
    [70, 0, 255],
    [0, 255, 159],
    [0, 216, 255],
    [0, 6, 255],
    [0, 63, 255],
    [31, 255, 0],
    [255, 57, 0],
    [255, 0, 210],
    [0, 255, 102],
    [242, 255, 0],
    [255, 191, 0],
    [0, 255, 63],
    [255, 0, 95],
    [146, 0, 255],
    [184, 255, 0],
    [255, 114, 0],
    [0, 255, 235],
    [255, 229, 0],
    [0, 178, 255],
    [255, 0, 114],
    [255, 0, 57],
    [0, 140, 255],
    [0, 121, 255],
    [12, 255, 0],
    [255, 210, 0],
    [0, 255, 44],
    [165, 255, 0],
    [0, 25, 255],
    [0, 255, 140],
    [0, 101, 255],
    [0, 255, 82],
    [223, 255, 0],
    [242, 0, 255],
    [89, 0, 255],
    [165, 0, 255],
    [70, 255, 0],
    [255, 0, 172],
    [255, 76, 0],
    [203, 255, 0],
    [204, 0, 255],
    [255, 0, 229],
    [255, 133, 0],
    [127, 0, 255],
    [0, 235, 255],
    [0, 255, 197],
    [255, 0, 191],
    [0, 44, 255],
    [50, 255, 0]
]


class ImageClass:
    "Stores the paths to images for a given class"

    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)


def get_color(label):
    """ Return a color from a set of predefined colors. Contains 80 colors in total.
    code originally from https://github.com/fizyr/keras-retinanet/
    Args
        label: The label to get the color for.
    Returns
        A list of three values representing a RGB color.
    """
    if label < len(COLORS):
        return COLORS[label]
    else:
        print('Label {} has no color, returning default.'.format(label))
        return (0, 255, 0)


def get_dataset(path, has_class_directories=True):
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp)
               if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(ImageClass(class_name, image_paths))

    return dataset


def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir, img) for img in images]
    return image_paths


class bounding_box:
    def __init__(self, xmin, ymin, xmax, ymax, c=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        self.c = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score


def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin

    union = w1 * h1 + w2 * h2 - intersect

    return float(intersect) / union


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3


def draw_boxes(image, boxes, labels, obj_thresh, quiet=True):
    """
    ### Draw bounding Boxes on image
    Parameters
    ----------
    image : [type]
        [description]
    boxes : [type]
        [description]
    labels : [type]
        [description]
    obj_thresh : [type]
        [description]
    quiet : [type], optional
        [description], by default True
    """
    for box in boxes:
        label_str = ''
        label = -1

        for i in range(len(labels)):
            if box.classes[i] > obj_thresh:
                if label_str != '':
                    label_str += ', '
                label_str += (labels[i] + ' ' + str(round(box.get_score() * 100, 2)) + '%')
                label = i
            if not quiet:
                print(label_str)

        if label >= 0:
            text_size = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 1.1e-3 * image.shape[0], 5)
            width, height = text_size[0][0], text_size[0][1]
            region = np.array([[box.xmin - 3, box.ymin],
                               [box.xmin - 3, box.ymin - height - 26],
                               [box.xmin + width + 13, box.ymin - height - 26],
                               [box.xmin + width + 13, box.ymin]], dtype='int32')

            cv2.rectangle(img=image,
                          pt1=(box.xmin, box.ymin),
                          pt2=(box.xmax, box.ymax),
                          color=get_color(label), thickness=5)
            cv2.fillPoly(img=image, pts=[region], color=get_color(label))
            cv2.putText(img=image,
                        text=label_str,
                        org=(box.xmin + 13, box.ymin - 13),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1e-3 * image.shape[0],
                        color=(0, 0, 0),
                        thickness=2)
    return image


def preprocess_input(image, net_h, net_w):
    new_h, new_w, _ = image.shape

    # determine the new size of the image
    if (float(net_w) / new_w) < (float(net_h) / new_h):
        new_h = (new_h * net_w) // new_w
        new_w = net_w
    else:
        new_w = (new_w * net_h) // new_h
        new_h = net_h

    # resize the image to the new size
    resized = cv2.resize(image[:, :, ::-1] / 255., (new_w, new_h))

    # embed the image into the standard letter box
    new_image = np.ones((net_h, net_w, 3)) * 0.5
    new_image[(net_h - new_h) // 2:(net_h + new_h) // 2, (net_w - new_w) // 2:(net_w + new_w) // 2, :] = resized
    new_image = np.expand_dims(new_image, 0)

    return new_image


def normalize(image):
    return image / 255.


def compute_overlap(a, b):
    """
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def crop(x, y, w, h, margin, img_width, img_height):
    xmin = int(x - w * margin)
    xmax = int(x + w * margin)
    ymin = int(y - h * margin)
    ymax = int(y + h * margin)
    if xmin < 0:
        xmin = 0
    if ymin < 0:
        ymin = 0
    if xmax > img_width:
        xmax = img_width
    if ymax > img_height:
        ymax = img_height
    return xmin, xmax, ymin, ymax
