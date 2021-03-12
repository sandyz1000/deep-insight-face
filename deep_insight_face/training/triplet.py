import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from ..handlers.evals import eval_callback
from ..networks.triplet import buildin_models, model_choice
from . import LOG_DIR, encoding_base
from collections import namedtuple
from ..datapipeline.generator import TripletGenerator

ImageDataPath = namedtuple("ImageDataPath", ("img_path", "pairs"))


class Train:
    def __init__(self,
                 model_path=None,
                 input_shape=(96, 96, 3),
                 emd_size=128,
                 mode=None,
                 logs_dir=LOG_DIR + os.path.sep + "triplet-logs") -> None:
        # Check training mode before initialization
        # assert mode in (model_choice.simple_triplet_nw, model_choice.semihard_triplet_nw), \
        #     "Invalid training function, other loss fn currently not supported"
        assert mode == model_choice.simple_triplet_nw, \
            "Invalid training function, other loss fn currently not supported"
        self.model_path = model_path
        self.input_shape = input_shape
        self.emd_size = emd_size
        self.logs_dir = logs_dir
        self.mode = mode
        facenet_weights = os.path.dirname(self.model_path) + os.sep + "bottleneck_facenet.h5"
        self.kwrg = {"summary": True, "use_keras_bn": True, "weights": facenet_weights,
                     "use_pretrained": False, "model_dir_path": self.model_path}
        self.model, self.bottleneck_model = self.load_model()

    def load_model(self):
        # Create a triplet n/w
        return buildin_models(self.mode, input_shape=self.input_shape, **self.kwrg)

    def __train_simpletriplet(self, train_data, batch_size, epochs, callbacks):

        x_train = TripletGenerator(
            train_data.img_path,
            train_data.pairs,
            horizontal_flip=True,
            shuffle=True,
            batch_size=batch_size,
            # use_multiprocessing=True,
            target_size=self.input_shape[:-1])

        history = self.model.fit(
            x_train,
            epochs=epochs,
            verbose=1,
            callbacks=callbacks
        )
        return history

    def __train_semihardtriplet(self, train_data, batch_size, epochs, callbacks):
        from keras.preprocessing.image import ImageDataGenerator
        x_train_gen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        x_train = x_train_gen.flow_from_directory(
            train_data.img_path,
            target_size=self.input_shape[:-1],
            class_mode='sparse',
            batch_size=batch_size)

        history = self.model.fit(
            x_train,
            epochs=epochs,
            steps_per_epoch=200,
            verbose=1,
            callbacks=callbacks
        )
        return history

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
             threshold=0.5):
        # val_binfile, val_image_path, val_pairs_txt = self.__filter_validation_data__(val_data)
        callbacks = [
            ModelCheckpoint(self.model_path),
            TensorBoard(log_dir=self.logs_dir),
            EarlyStopping(monitor='loss', patience=5),
            # eval_callback(self.model, val_binfile, image_path=val_image_path,
            #               pairs_txt=val_pairs_txt, batch_size=batch_size)
        ]
        if self.mode == model_choice.simple_triplet_nw:
            history = self.__train_simpletriplet(train_data, batch_size, epochs, callbacks)
        else:
            history = self.__train_semihardtriplet(train_data, batch_size, epochs, callbacks)

        self.bottleneck_model.save_weights(self.model_path)
        self.save_training(history)

    def save_training(self, history):
        plt.figure(figsize=(8, 8))
        plt.plot(history.history['loss'], label='training loss')
        # plt.plot(history.history['val_loss'], label='validation loss')
        plt.legend()
        plt.title('Train/validation loss')
        plt.savefig("triplet_loss.png")

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
        encoding = img_to_encoding(self.model)(image_path)

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


class img_to_encoding(encoding_base):
    def __init__(self, emd_model, img_size=(96, 96)) -> None:
        assert len(img_size) == 2, "Invalid Image size format"
        super(img_to_encoding, self).__init__(emd_model, img_size)

    def _embedding(self, image: np.ndarray) -> np.ndarray:
        img = cv2.resize(image, tuple(self.img_size), interpolation=Image.BICUBIC) / 255.
        inp = np.expand_dims(img, axis=0)
        embedding = self.emd_model.predict_on_batch(inp)
        return embedding

