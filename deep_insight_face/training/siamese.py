import os
import matplotlib.pyplot as plt
from . import LOG_DIR
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from ..networks.siamese import buildin_models
from collections import namedtuple
from ..datagen.generator import facematch_datagenerator

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
        ]

        x_train = facematch_datagenerator(
            train_data.img_path,
            train_data.pairs,
            batch_size=batch_size,
            target_size=self.input_shape[:-1]
        )

        history = self.model.fit(
            x_train,
            batch_size=batch_size,
            steps_per_epoch=100,
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

