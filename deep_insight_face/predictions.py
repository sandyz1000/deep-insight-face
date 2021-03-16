
import numpy as np
from abc import ABCMeta, abstractmethod
import sys
import cv2
import importlib
from PIL import Image
from typing import Any, Union
from six import add_metaclass
from keras.applications.vgg16 import preprocess_input


@add_metaclass(ABCMeta)
class encoding_base:
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
    # TODO: Fix this module
    # package = sys.modules[__name__].__package__
    # mod = importlib.import_module(f"{package}.{name}")
    # _cls = getattr(mod, "img_to_encoding")
    emd = _cls(model, img_size=image_size[:-1])._embedding(img_path)
    return emd


class SiamesePrediction(encoding_base):
    def __init__(self, emd_model, img_size=(112, 112)):
        assert len(img_size) == 2, "Invalid Image size format"
        super(SiamesePrediction, self).__init__(emd_model, img_size)

    def verify(self, image_path, identity, database, threshold=0.3):
        """
        Function that verifies if the person on the "image_path" image is "identity".

        Arguments:
        ----------
        image_path -- path to an image
        identity -- string, name of the person you'd like to verify the identity. 
        Has to be a resident of the Happy house.
        database -- python dictionary mapping names of allowed people's names (strings) to their encodings (vectors).
        model -- your Inception model instance in Keras

        Returns:
        --------
        dist -- distance between the image_path and the image of "identity" in the database.
        door_open -- True, if the door should open. False otherwise.
        """
        # TODO: Fix this method
        # Step 1: Compute the encoding for the image. Use img_to_encoding() see example above. (≈ 1 line)
        encoding = self._embedding(image_path)

        # Step 2: Compute distance with identity's image (≈ 1 line)
        input_pairs = []
        x = database[identity]
        for i in range(len(x)):
            input_pairs.append([encoding, x[i]])
        input_pairs = np.array(input_pairs)
        dist = np.average(self.model.predict([input_pairs[:, 0], input_pairs[:, 1]]), axis=-1)[0]

        # Step 3: Open the door if dist < threshold, else don't open (≈ 3 lines)
        if dist < threshold:
            print("It's " + str(identity))
            is_valid = True
        else:
            print("It's not " + str(identity))
            is_valid = False

        return dist, is_valid

    def _embedding(self, image: np.ndarray, rescale: float = 1 / 255.) -> np.ndarray:
        assert isinstance(image, np.ndarray), "Invalid image format, should be of type numpy array"
        img = cv2.resize(image, tuple(self.img_size), interpolation=Image.BICUBIC) * rescale
        inp = np.expand_dims(img, axis=0)
        inp = preprocess_input(inp)
        return self.emd_model.predict_on_batch(inp)


class TripletPrediction(encoding_base):
    def __init__(self, emd_model, img_size=(96, 96)):
        assert len(img_size) == 2, "Invalid Image size format"
        super(TripletPrediction, self).__init__(emd_model, img_size)

    def verify(self, image_path, identity, database, threshold=0.7):
        """
        Function that verifies if the person on the "image_path" image is "identity".=

        Arguments:
        ----------
        image_path -- path to an image
        identity -- string, name of the person you'd like to verify the identity. Has to be a resident of the Happy house.
        database -- python dictionary mapping names of allowed people's names (strings) to their encodings (vectors).
        model -- your Inception model instance in Keras

        Returns:
        --------
        dist -- distance between the image_path and the image of "identity" in the database.
        door_open -- True, if the door should open. False otherwise.
        """
        # TODO: Fix this method

        # Step 1: Compute the encoding for the image. Use img_to_encoding() see example above. (≈ 1 line)
        encoding = self._embedding(image_path)

        # Step 2: Compute distance with identity's image (≈ 1 line)
        dist = float(np.linalg.norm(encoding - database[identity]))

        # Step 3: Open the door if dist < threshold, else don't open (≈ 3 lines)
        if dist < threshold:
            print("It's " + str(identity))
            is_valid = True
        else:
            print("It's not " + str(identity))
            is_valid = False

        # model_dir_path = './demo/models'
        # image_dir_path = "./demo/data/test-images"

        # fnet.load_model(model_dir_path)

        # enc1 = fnet.img_to_encoding(os.path.join(
        #     image_dir_path, "3000CD0118870/4016_3000CD0118870_CANCELLED_CNCLD_APPLICANT-PHOTO.jpg"))

        # enc2 = fnet.img_to_encoding(os.path.join(
        #     image_dir_path, "3000CD0118870/4016_3000CD0118870_CANCELLED_CNCLD_VOTER.jpg"))

        # distance = np.linalg.norm(enc1 - enc2)
        # print(distance)

        return dist, is_valid

    def _embedding(self, image: np.ndarray, rescale: float = 1 / 255.) -> np.ndarray:
        assert isinstance(image, np.ndarray), "Invalid image format, should be of type numpy array"
        img = cv2.resize(image, tuple(self.img_size), interpolation=Image.BICUBIC) * rescale
        inp = np.expand_dims(img, axis=0)
        return self.emd_model.predict_on_batch(inp)
