
import os
import numpy as np
from abc import ABCMeta, abstractmethod
import sys
import importlib
from PIL import Image
from typing import Any, Union

BASE_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir)
)

LOG_DIR = os.path.join(BASE_DIR, "logs")


class encoding_base(metaclass=ABCMeta):
    """ Singleton Abstract base for encoding image
    """

    def __init__(self, emd_model, img_size=(96, 96)):
        self.emd_model = emd_model
        self.img_size = img_size

    _instances = {}

    def read_image(self, img_path) -> np.ndarray:
        im = Image.open(img_path)
        return np.array(im, dtype=np.uint8)
    
    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(encoding_base, cls).__new__(cls)
        return cls._instances[cls]

    @abstractmethod
    def _embedding(self, image: Union[str, np.ndarray]):
        pass


def get_embedding(name: str, model: Any, img_path: str, image_size=(112, 112, 3)) -> np.ndarray:
    package = sys.modules[__name__].__package__
    mod = importlib.import_module(f"{package}.{name}")
    _cls = getattr(mod, "img_to_encoding")
    emd = _cls(model, img_size=image_size[:-1])._embedding(img_path)
    return emd
