import os
import uuid
import cv2
import numpy as np
import colorsys
from .utility import (preprocess_input, decode_netout, draw_boxes,
                      correct_yolo_boxes, do_nms, thread_safe_singleton)
from tensorflow.python import keras
from PIL import Image
import warnings
from tempfile import NamedTemporaryFile
warnings.simplefilter('ignore')


class YoloArgs(object):
    def __init__(self,
                 yolo_model, anchors, classes,
                 obj_thresh=0.3, nms_thresh=0.45, img_size=(416, 416)):
        self.model = yolo_model
        self.anchors = anchors
        self.classes = classes
        self.obj_thresh = obj_thresh
        self.nms_thresh = nms_thresh
        self.img_size = img_size


@thread_safe_singleton
class YOLO(object):
    def __init__(self, args: YoloArgs) -> None:
        self.model_path = args.model
        self.class_names = args.classes
        self.anchors = args.anchors
        self.net_h, self.net_w = args.img_size[0], args.img_size[1]
        self.obj_thresh = args.obj_thresh
        self.nms_thresh = args.nms_thresh
        self.__load_model__()
        print("======= YoLov3 weight load complete ======= ")

    def __load_model__(self):
        self.model = keras.models.load_model(self.model_path)

    def letterbox_image(self, image, size):
        """Resize image with unchanged aspect ratio using padding
        """
        img_width, img_height = image.size
        w, h = size
        scale = min(w / img_width, h / img_height)
        nw = int(img_width * scale)
        nh = int(img_height * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        return new_image

    def detect_boxes(self, image, draw=True):
        """ Predict and get bounding box using YOLOv3 model
        """
        # new_image_size = (image.width - (image.width % 32),
        #                   image.height - (image.height % 32))
        # boxed_image = self.letterbox_image(image, new_image_size)
        img_size = (image.width, image.height)
        image = np.array(image, dtype='float32')
        image_h, image_w, _ = image.shape
        nb_images = 1
        batch_input = np.zeros((nb_images, self.net_h, self.net_w, 3))
        # preprocess the input
        for i in range(nb_images):
            batch_input[i] = preprocess_input(image, self.net_h, self.net_w)

        # run the prediction
        batch_output = self.model.predict_on_batch(batch_input)
        batch_boxes = [None] * nb_images

        for i in range(nb_images):
            yolos = [batch_output[0][i], batch_output[1][i], batch_output[2][i]]
            boxes = []
            # decode the output of the network
            for j in range(len(yolos)):
                yolo_anchors = self.anchors[(2 - j) * 6:(3 - j) * 6]
                boxes += decode_netout(yolos[j], yolo_anchors, self.obj_thresh, self.net_h, self.net_w)

            # correct the sizes of the bounding boxes
            correct_yolo_boxes(boxes, image_h, image_w, self.net_h, self.net_w)
            # suppress non-maximal boxes
            do_nms(boxes, self.nms_thresh)
            batch_boxes[i] = boxes

        out = []
        for box in batch_boxes:
            if draw:
                with NamedTemporaryFile(delete=False, prefix="facematch_detect") as f:
                    f.name += ".png"
                    dimg = draw_boxes(image, box, self.class_names, self.obj_thresh)
                    gray = cv2.cvtColor(dimg, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(f.name, gray)
            out.append(self.transform_format(box, img_size))

        return image, out[0]

    def transform_format(self, boxes, img_size):
        final_boxes = []
        for _, box in enumerate(boxes):
            label = -1
            for i in range(len(self.class_names)):
                if box.classes[i] > self.obj_thresh:
                    label = i
            if label >= 0:
                left, top, right, bottom = box.xmin, box.ymin, box.xmax, box.ymax
                top = max(0, np.floor(top - 0.5).astype('int32'))
                left = max(0, np.floor(left - 0.5).astype('int32'))
                bottom = min(img_size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(img_size[0], np.floor(right + 0.5).astype('int32'))
                final_boxes.append([left, top, right, bottom])

                predicted_class = box.get_label()
                score = box.get_score()
                _ = '{} {:.2f}'.format(predicted_class, score)
                # draw = ImageDraw.Draw(image)
                # print(text, (left, top), (right, bottom))
        return final_boxes

    def generate_colormap(self):
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        np.random.shuffle(self.colors)
