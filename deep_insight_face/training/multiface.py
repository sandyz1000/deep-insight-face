import os
import cv2
from tensorflow import keras
import numpy as np
import tensorflow as tf
from ..handlers import losses, myCallbacks
from ..handlers.evals import eval_callback
from . import encoding_base
from ..networks.multiface import NormDense, buildin_models
from ..datapipeline.data import prepare_dataset, Triplet_dataset
from ..backbones import mobile_facenet
import tensorflow_addons as tfa


class Train:
    def __init__(
        self,
        data_path,
        save_path,
        eval_paths=[],
        basic_model=None,
        model=None,
        compile=True,
        batch_size=128,
        lr_base=0.001,
        lr_decay=0.05,  # lr_decay < 1 for exponential, or it's cosine decay_steps
        lr_on_batch=0,  # lr_on_batch < 1 for update lr on epoch, or update on every [NUM] batches
        lr_min=0,
        eval_freq=1,
        random_status=0,
        custom_objects={},
        log_dir="logs"
    ):
        custom_objects.update(
            {
                "NormDense": NormDense,
                "margin_softmax": losses.margin_softmax,
                "arcface_loss": losses.arcface_loss,
                "ArcfaceLoss": losses.ArcfaceLoss,
                "CenterLoss": losses.CenterLoss,
                "batch_hard_triplet_loss": losses.batch_hard_triplet_loss,
                "batch_all_triplet_loss": losses.batch_all_triplet_loss,
                "BatchHardTripletLoss": losses.BatchHardTripletLoss,
                "BatchAllTripletLoss": losses.BatchAllTripletLoss,
                "logits_accuracy": self.logits_accuracy,
            }
        )
        self.model, self.basic_model, self.save_path = None, None, save_path
        if isinstance(model, str):
            if model.endswith(".h5") and os.path.exists(model):
                print(">>>> Load model from h5 file: %s..." % model)
                with keras.utils.custom_object_scope(custom_objects):
                    self.model = keras.models.load_model(model, compile=compile, custom_objects=custom_objects)
                basic_model = basic_model if basic_model is not None else self.__search_embedding_layer__(self.model)
                self.basic_model = keras.models.Model(self.model.inputs[0], self.model.layers[basic_model].output)
                self.model.summary()
        elif isinstance(model, keras.models.Model):
            self.model = model
            basic_model = basic_model if basic_model is not None else self.__search_embedding_layer__(self.model)
            self.basic_model = keras.models.Model(self.model.inputs[0], self.model.layers[basic_model].output)
        elif isinstance(basic_model, str):
            if basic_model.endswith(".h5") and os.path.exists(basic_model):
                print(">>>> Load basic_model from h5 file: %s..." % basic_model)
                with keras.utils.custom_object_scope(custom_objects):
                    self.basic_model = keras.models.load_model(
                        basic_model, compile=compile, custom_objects=custom_objects)
        elif isinstance(basic_model, keras.models.Model):
            self.basic_model = basic_model

        if self.basic_model is None:
            print(
                "Initialize model by:\n"
                "| basic_model                                                     | model           |\n"
                "| --------------------------------------------------------------- | --------------- |\n"
                "| model structure                                                 | None            |\n"
                "| basic model .h5 file                                            | None            |\n"
                "| None for 'embedding' layer or layer index of basic model output | model .h5 file  |\n"
                "| None for 'embedding' layer or layer index of basic model output | model structure |\n"
            )
            return

        self.softmax, self.arcface, self.triplet = "softmax", "arcface", "triplet"

        self.batch_size = batch_size
        if tf.distribute.has_strategy():
            strategy = tf.distribute.get_strategy()
            self.batch_size = batch_size * strategy.num_replicas_in_sync
            print(">>>> num_replicas_in_sync: %d, batch_size: %d" % (strategy.num_replicas_in_sync, self.batch_size))
        self.data_path, self.random_status = data_path, random_status
        self.train_ds, self.steps_per_epoch, self.classes = None, None, 0
        self.is_triplet_dataset = False
        self.default_optimizer = "adam"
        self.metrics = ["accuracy"]
        my_evals = [eval_callback(self.basic_model, ii, batch_size=batch_size,
                                  eval_freq=eval_freq) for ii in eval_paths]
        if len(my_evals) != 0:
            my_evals[-1].save_model = os.path.splitext(save_path)[0]
        basic_callbacks = myCallbacks.basic_callbacks(
            checkpoint=os.path.dirname(save_path), log_dir=log_dir,
            evals=my_evals, lr=lr_base,
            lr_decay=lr_decay, lr_min=lr_min, lr_on_batch=lr_on_batch
        )
        self.my_evals = my_evals
        self.basic_callbacks = basic_callbacks
        self.my_hist = [ii for ii in self.basic_callbacks if isinstance(ii, myCallbacks.My_history)][0]
        # NOTE: Comment while executing program in production, args: None or triplet
        # self.__init_dataset__('triplet')

    def __search_embedding_layer__(self, model):
        for ii in range(1, 6):
            if model.layers[-ii].name == 'embedding':
                return -ii

    def __init_dataset__(self, type):
        if type == self.triplet:
            if self.train_ds is None or not self.is_triplet_dataset:
                print(">>>> Init triplet dataset...")
                # batch_size = int(self.batch_size / 4 * 1.5)
                batch_size = self.batch_size // 4
                tt = Triplet_dataset(self.data_path, batch_size=batch_size,
                                     random_status=self.random_status, random_crop=(100, 100, 3))
                self.train_ds = tt.train_dataset
                self.classes = self.train_ds.element_spec[-1].shape[-1]
                self.is_triplet_dataset = True
        else:
            if self.train_ds is None or self.is_triplet_dataset:
                print(">>>> Init softmax dataset...")
                self.train_ds = prepare_dataset(
                    self.data_path, batch_size=self.batch_size, random_status=self.random_status, random_crop=(
                        100, 100, 3)
                )
                self.classes = self.train_ds.element_spec[-1].shape[-1]
                self.is_triplet_dataset = False

    def __init_optimizer__(self, optimizer):
        if optimizer is None:
            if self.model is not None and self.model.optimizer is not None:
                # Model loaded from .h5 file already compiled
                self.optimizer = self.model.optimizer
            else:
                self.optimizer = self.default_optimizer
        else:
            self.optimizer = optimizer

    def __init_model__(self, type):
        inputs = self.basic_model.inputs[0]
        embedding = self.basic_model.outputs[0]
        while self.model is not None and isinstance(self.model.layers[-1], keras.layers.Concatenate):
            # In case of centerloss or concatenated triplet model
            self.model = keras.models.Model(inputs, self.model.layers[-2].output)

        if type == self.softmax:
            if self.model is None or self.model.output_names[-1] != self.softmax:
                print(">>>> Add softmax layer...")
                output = keras.layers.Dense(self.classes, name=self.softmax, activation="softmax")(embedding)
                self.model = keras.models.Model(inputs, output)
        elif type == self.arcface:
            if self.model is None or self.model.output_names[-1] != self.arcface:
                print(">>>> Add arcface layer...")
                output = NormDense(self.classes, name=self.arcface)(embedding)
                self.model = keras.models.Model(inputs, output)
        elif type == self.triplet:
            self.model = self.basic_model
        else:
            print("What do you want!!!")

    def __init_type_by_loss__(self, loss):
        print(">>>> Init type by loss function name...")
        if loss.__class__.__name__ == "function":
            ss = loss.__name__.lower()
        else:
            ss = loss.__class__.__name__.lower()
        if self.softmax in ss or ss == "categorical_crossentropy":
            return self.softmax
        elif self.arcface in ss:
            return self.arcface
        elif self.triplet in ss:
            return self.triplet
        else:
            return self.softmax

    def __basic_train__(self, loss, epochs, initial_epoch=0):
        """
        Method to be invoke by used to initialize training
        Arguments:
            loss {[type]} -- [description]
            epochs {[type]} -- [description]

        Keyword Arguments:
            initial_epoch {[type]} -- [description] (default: {0})
        """
        self.model.compile(optimizer=self.optimizer, loss=loss, metrics=self.metrics)
        self.model.fit(
            self.train_ds,
            epochs=epochs,
            verbose=1,
            callbacks=self.callbacks,
            initial_epoch=initial_epoch,
            steps_per_epoch=self.steps_per_epoch,
            use_multiprocessing=True,
            workers=4,
        )

    def logits_accuracy(self, y_true, y_pred):
        """ Accuracy function for logits only """
        # tf.print(y_true.shape, y_pred.shape)
        # labels = tf.one_hot(tf.squeeze(y_true), depth=classes, dtype=tf.int32)
        return tf.keras.metrics.categorical_accuracy(y_true, y_pred[:, -self.classes:])

    def train(self, train_schedule, initial_epoch=0):
        for sch in train_schedule:
            if sch.get("loss", None) is None:
                continue
            cur_loss = sch["loss"]
            self.basic_model.trainable = True
            self.__init_optimizer__(sch.get("optimizer", None))

            if isinstance(cur_loss, losses.TripletLossWapper) and cur_loss.logits_loss is not None:
                type = sch.get("type", None) or self.__init_type_by_loss__(cur_loss.logits_loss)
                cur_loss.feature_dim = self.basic_model.output_shape[-1]
                print(">>>> Train Triplet + %s, feature_dim = %d ..." % (type, cur_loss.feature_dim))
                self.__init_dataset__(self.triplet)
                self.__init_model__(type)
                self.model = keras.models.Model(
                    self.model.inputs[0], keras.layers.concatenate(
                        [self.basic_model.outputs[0], self.model.outputs[-1]])
                )
                type = self.triplet + " + " + type
            else:
                type = sch.get("type", None) or self.__init_type_by_loss__(cur_loss)
                print(">>>> Train %s..." % type)
                self.__init_dataset__(type)
                self.__init_model__(type)

            if sch.get("centerloss", False):
                print(">>>> Train centerloss...")
                center_loss = cur_loss
                if not isinstance(center_loss, losses.CenterLoss):
                    feature_dim = self.basic_model.output_shape[-1]
                    # initial_file = self.basic_model.name + "_centers.npy"
                    initial_file = os.path.splitext(self.save_path)[0] + "_centers.npy"
                    logits_loss = cur_loss
                    center_loss = losses.CenterLoss(
                        self.classes, feature_dim=feature_dim, factor=1.0,
                        initial_file=initial_file, logits_loss=logits_loss
                    )
                    cur_loss = center_loss
                    # self.my_hist.custom_obj["centerloss"] = lambda : cur_loss.centerloss
                self.model = keras.models.Model(
                    self.model.inputs[0], keras.layers.concatenate(
                        [self.basic_model.outputs[0], self.model.outputs[-1]])
                )
                self.callbacks = self.my_evals + [center_loss.save_centers_callback] + self.basic_callbacks
            else:
                self.callbacks = self.my_evals + self.basic_callbacks
            self.metrics = None if type == self.triplet else [self.logits_accuracy]

            if sch.get("bottleneckOnly", False):
                print(">>>> Train bottleneckOnly...")
                self.basic_model.trainable = False
                self.callbacks = self.callbacks[len(self.my_evals):]  # Exclude evaluation callbacks
                self.__basic_train__(cur_loss, sch["epoch"], initial_epoch=0)
                self.basic_model.trainable = True
            else:
                self.__basic_train__(cur_loss, initial_epoch + sch["epoch"], initial_epoch=initial_epoch)
                initial_epoch += sch["epoch"]

            print(
                ">>>> Train %s DONE!!! epochs = %s, model.stop_training = %s"
                % (type, self.model.history.epoch, self.model.stop_training)
            )
            print(">>>> My history:")
            self.my_hist.print_hist()
            if self.model.stop_training:
                print(">>>> But it's an early stop, break... >>>>")
                break
            print(">>>> ================== >>>>")


class img_to_encoding(encoding_base):
    def __init__(self, emd_model, img_size=(96, 96)) -> None:
        assert len(img_size) == 2, "Invalid Image size format"
        super(img_to_encoding, self).__init__(emd_model, img_size)

    def _embedding(self, image: np.ndarray) -> np.ndarray:
        img = cv2.resize(image, self.img_size, interpolation=cv2.INTER_AREA)
        img = np.around(img / 255.0, decimals=12)
        x_train = np.array([img])
        embedding = self.emd_model.predict_on_batch(x_train)
        return embedding


# --------------------------------
# Training Conf for DEEPINSIGHT
# --------------------------------
def multiface_train(data_path, model_path, eval_paths=[], batch_size=512, log_dir="./logs"):
    train_ds = prepare_dataset(data_path, batch_size=batch_size, random_status=3, random_crop=(100, 100, 3))
    classes = train_ds.element_spec[-1].shape[-1]
    # Model
    basic_model = mobile_facenet.mobile_facenet(256, dropout=0, name="mobile_facenet_256")
    model_output = keras.layers.Dense(classes, activation="softmax")(basic_model.outputs[0])
    model = keras.models.Model(basic_model.inputs[0], model_output)
    # Evals and basic callbacks
    my_evals = [eval_callback(basic_model, ii, batch_size=512, eval_freq=1) for ii in eval_paths]
    my_evals[-1].save_model = model_path
    basic_callbacks = myCallbacks.basic_callbacks(checkpoint=os.path.dirname(model_path),
                                                  log_dir=log_dir,
                                                  evals=my_evals, lr=0.001)
    callbacks = my_evals + basic_callbacks
    # Compile and fit
    model.compile(optimizer='nadam',
                  loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                  metrics=["accuracy"])
    model.fit(train_ds, epochs=15, callbacks=callbacks, verbose=1)


def deepinsight_train(name, data_path, model_path, eval_paths, emd_shape=256, batch_size=128, log_dir="./logs"):
    # basic_model = train.buildin_models("ResNet101V2", dropout=0, emb_shape=512)
    # basic_model = train.buildin_models("ResNest101", dropout=0, emb_shape=512)
    # basic_model = train.buildin_models('EfficientNetB0', dropout=0, emb_shape=256)
    # basic_model = train.buildin_models('EfficientNetB4', dropout=0, emb_shape=256)
    # basic_model = mobile_facenet.mobile_facenet(256, dropout=0, name="mobile_facenet_256")
    # basic_model = mobile_facenet.mobile_facenet(256, dropout=0, name="se_mobile_facenet_256", use_se=True)
    basic_model = buildin_models(name, dropout=0, emb_shape=emd_shape)
    tt = Train(
        data_path, save_path=model_path, eval_paths=eval_paths,
        basic_model=basic_model, compile=True, log_dir=log_dir, lr_base=0.001, batch_size=batch_size, random_status=3)
    optimizer = tfa.optimizers.AdamW(weight_decay=5e-5)
    sch = [
        {"loss": keras.losses.CategoricalCrossentropy(
            label_smoothing=0.1), "centerloss": 1, "optimizer": optimizer, "epoch": 20},
        {"loss": keras.losses.CategoricalCrossentropy(label_smoothing=0.1), "centerloss": 32, "epoch": 20},
        {"loss": keras.losses.CategoricalCrossentropy(label_smoothing=0.1), "centerloss": 64, "epoch": 20},
        {"loss": losses.ArcfaceLoss(), "epoch": 20, "triplet": 64, "alpha": 0.3},
        {"loss": losses.ArcfaceLoss(), "epoch": 20, "triplet": 64, "alpha": 0.25},
        {"loss": losses.ArcfaceLoss(), "epoch": 20, "triplet": 64, "alpha": 0.2},
    ]
    tt.train(sch, 0)


def deepinsight_train2(name, data_path, model_path, eval_paths, emd_shape=128, batch_size=512, log_dir="./logs"):
    basic_model = buildin_models(name, dropout=0, emb_shape=emd_shape)
    tt = Train(
        data_path, save_path=model_path, eval_paths=eval_paths,
        basic_model=basic_model, compile=True, log_dir=log_dir, lr_base=0.001, batch_size=batch_size, random_status=3)
    sch = [
        {"loss": keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
         "centerloss": True, "optimizer": "nadam", "epoch": 25},
        {"loss": keras.losses.CategoricalCrossentropy(
            label_smoothing=0.1), "centerloss": True, "optimizer": "nadam", "epoch": 6},
        # {"loss": losses.scale_softmax, "epoch": 10},
        {"loss": losses.ArcfaceLoss(), "centerloss": True, "epoch": 35},
        {"loss": losses.BatchHardTripletLoss(0.35), "epoch": 10},
        {"loss": losses.BatchAllTripletLoss(0.33), "epoch": 10},
    ]
    tt.train(sch, 0)
