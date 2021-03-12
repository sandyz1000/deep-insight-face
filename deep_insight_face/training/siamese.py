import os
import cv2
import numpy as np
from typing import Union
import matplotlib.pyplot as plt
from PIL import Image
from . import LOG_DIR, encoding_base
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from ..networks.siamese import buildin_models
from collections import namedtuple
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import img_to_array
from ..datapipeline.generator import SiameseDataGenerator

ImageDataPath = namedtuple("ImageDataPath", ("img_path", "pairs"))


class Train:
    def __init__(self,
                 model_path=None,
                 input_shape=(112, 112, 3),
                 emd_size=128,
                 logs_dir=LOG_DIR + os.path.sep + "siamese-logs") -> None:
        self.model_path = model_path
        self.input_shape = input_shape
        self.emd_size = emd_size
        self.model = None
        self.logs_dir = logs_dir
        self.kwrg = {"summary": True, "model_path": self.model_path}
        self.model, self.bottleneck_model = self.load_model()

    def load_model(self):
        # Create siamese n/w
        return buildin_models(emd_size=self.emd_size, input_shape=self.input_shape, **self.kwrg)

    def __filter_validation_data__(self, valid_data):
        val_binfile, val_image_path, val_pairs_txt = None, None, None
        if valid_data and valid_data.endswith(".bin"):
            val_binfile = valid_data
        if valid_data and isinstance(valid_data, ImageDataPath):
            val_image_path, val_pairs_txt = valid_data.img_path, valid_data.pairs
        return val_binfile, val_image_path, val_pairs_txt

    def _fit(self,
             train_data,
             val_data=None,  # bin file or ImageDataPath
             epochs=20,
             batch_size=128,
             threshold=0.4):
        # val_binfile, val_image_path, val_pairs_txt = self.__filter_validation_data__(val_data)
        callbacks = [
            ModelCheckpoint(self.model_path), 
            TensorBoard(log_dir=self.logs_dir),
            EarlyStopping(monitor='loss', patience=5),
            # eval_callback(self.model, val_binfile, image_path=val_image_path,
            #               pairs_txt=val_pairs_txt, batch_size=batch_size)
        ]

        x_train = SiameseDataGenerator(
            train_data.img_path,
            train_data.pairs,
            horizontal_flip=True,
            vertical_flip=True,
            shuffle=True,
            batch_size=batch_size,
            target_size=self.input_shape[:-1])

        history = self.model.fit(
            x_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            callbacks=callbacks)

        self.bottleneck_model.save_weights(self.model_path)
        self.save_training(history)

    def save_training(self, history):
        plt.figure(figsize=(8, 8))
        plt.plot(history.history['loss'], label='training loss')
        plt.plot(history.history['val_loss'], label='validation loss')
        plt.legend()
        plt.title('Train/validation loss')
        plt.savefig("siamese_loss.png")

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
        encoding = img_to_encoding(self.model)(image_path)

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


class img_to_encoding(encoding_base):
    def __init__(self, emd_model, img_size=(112, 112)):
        assert len(img_size) == 2, "Invalid Image size format"
        super(img_to_encoding, self).__init__(emd_model, img_size)

    def _embedding(self, image: Union[str, np.ndarray]) -> np.ndarray:
        if isinstance(image, str) and os.path.exists(image):
            image = cv2.imread(image, cv2.IMREAD_COLOR)[..., ::-1]  # Use RGB Format
        img = cv2.resize(image, tuple(self.img_size), interpolation=Image.BICUBIC) / 255.
        inp = np.expand_dims(img, axis=0)
        inp = preprocess_input(inp)
        return self.emd_model.predict_on_batch(inp)

