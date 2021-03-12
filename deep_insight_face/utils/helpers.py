import cv2
import numpy as np
from PIL import ImageFile, Image


class ImageReadError(ValueError):
    pass


def concat_images(X):
    """Concatenates a bunch of images into a big matrix for plotting purposes."""
    nc, h, w, _ = X.shape
    X = X.reshape(nc, h, w)
    n = np.ceil(np.sqrt(nc)).astype("int8")
    img = np.zeros((n * w, n * h))
    x = 0
    y = 0
    for example in range(nc):
        img[x * w:(x + 1) * w, y * h:(y + 1) * h] = X[example]
        y += 1
        if y >= n:
            y = 0
            x += 1
    return img


def load_image(img_file, target_size=None):
    from tensorflow import keras

    if target_size is not None:
        return np.asarray(keras.preprocessing.image.load_img(img_file, target_size=target_size))

    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # OpenCV loads images with color channels in BGR order.
    # So we need to reverse them
    return img[..., ::-1]


def read_image_pil(img_path, mode='RGB'):
    """
    Loads an image file (.jpg, .png, etc) into a numpy array

    :param file: image file name or file object to load
    :param mode: format to convert the image to. Only 'RGB' (8-bit RGB, 3 channels) and 'L'
    (black and white) are supported.
    :return: image contents as numpy array
    """
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        im = Image.open(img_path)
        if mode:
            im = im.convert(mode)
        return np.array(im)
    except Exception as e:
        raise ValueError("Image is empty" + str(e))


def read_img_cv2(img_path, mode='RGB'):
    """
    Loads an image file (.jpg, .png, etc) into a numpy array, for grey scale image mode=LA
    :return: image contents as numpy array
    """
    try:
        im = cv2.imread(img_path)
        mode = cv2.COLOR_BGR2GRAY if mode == 'gray' else cv2.COLOR_BGR2RGB
        im = cv2.cvtColor(im, mode)
        return im
    except Exception as e:
        raise ImageReadError("Image is empty" + str(e))
