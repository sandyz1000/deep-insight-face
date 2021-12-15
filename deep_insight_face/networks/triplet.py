
import enum
import typing
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, ResNet50V2, VGG16
from keras import optimizers as KO
from keras.layers import Conv2D, Input, DepthwiseConv2D, Dropout, PReLU
import keras.layers as KL
from keras.models import Model, Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Lambda, Flatten, Dense
from keras import backend as K
from .inceptionv3 import InceptionNetwork


def triplet_loss(y_true, y_pred, alpha=0.4):
    """
    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """
    # print('y_pred.shape = ', y_pred)

    _len = y_pred.shape.as_list()[-1]

    anchor = y_pred[:, 0:int(_len * 1 / 3)]
    positive = y_pred[:, int(_len * 1 / 3):int(_len * 2 / 3)]
    negative = y_pred[:, int(_len * 2 / 3):int(_len * 3 / 3)]

    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor - positive), axis=1)

    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor - negative), axis=1)

    # compute loss
    basic_loss = pos_dist - neg_dist + alpha
    loss = K.maximum(basic_loss, 0.0)

    return loss


def triplet_loss_test():
    tf.set_random_seed(1)
    y_true = (None, None, None)
    y_pred = (tf.random_normal([3, 128], mean=6, stddev=0.1, seed=1),
              tf.random_normal([3, 128], mean=1, stddev=1, seed=1),
              tf.random_normal([3, 128], mean=3, stddev=4, seed=1))
    loss = triplet_loss(y_true, y_pred)

    print("loss = " + str(loss.eval()))


class bottleneck_network:
    """
    ------------------------------
    FaceNet Triplet configuration
    ------------------------------
    Return faceRecognition model based on various network

    Keyword Arguments:
        model_dir_path {[type]} -- [description] (default: {None})
        emd_size {[type]} -- [description] (default: {128})
        input_shape {[type]} -- [description] (default: {(96, 96, 3)})
    """

    def __init__(self,
                 net: str = "resnet",
                 emd_size: int = 128,
                 input_shape: typing.Tuple = (96, 96, 3), **kwargs):
        assert net in ('mobilenet', 'resnet', 'vgg16'), "Invalid bottleneck network"
        self.net = net
        self.emd_size = emd_size
        self.input_shape = input_shape
        self.kwargs = kwargs

    def __call__(self, default_model_ver='v1', dropout=.2) -> typing.Any:
        base_model = getattr(self, 'build_models_' + default_model_ver)(dropout=dropout)
        return base_model

    def __get_bottleneck(self, ):
        if self.net == "mobilenet":
            bottleneck = MobileNetV2(input_shape=self.input_shape, include_top=False, weights='imagenet')
        elif self.net == "resnet":
            bottleneck = ResNet50V2(input_shape=self.input_shape, include_top=False, weights='imagenet')
        elif self.net == "vgg16":
            bottleneck = VGG16(include_top=False, weights='imagenet', input_shape=self.input_shape)
        elif self.net == "inception":
            bottleneck = InceptionNetwork(
                input_shape=self.input_shape,
                emd_size=self.emd_size,
                weights=self.kwargs.get('weights', None)
            )
        return bottleneck

    def build_models_v1(self, dropout=0.3):
        seq_layers = [self.__get_bottleneck()]
        seq_layers += [
            KL.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'),
            KL.MaxPooling2D(pool_size=2),
            KL.Dropout(0.3),
            KL.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'),
            KL.MaxPooling2D(pool_size=2),
            KL.Dropout(dropout),
            KL.Flatten(name='flatten'),
            KL.Dense(self.emd_size, activation=None, name="embeddings"),  # No activation on final dense layer
            # KL.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))  # L2 normalize embeddings
        ]
        base_model = Sequential(seq_layers)
        # base_model.layers[0].trainable = False
        return base_model

    def build_models_v2(self, dropout=0.3):
        xx = self.__get_bottleneck()

        nw_input = xx.inputs[0]
        nw_output = xx.outputs[0]
        xx.trainable = True

        nn = KL.Conv2D(512, 1, use_bias=False)(nw_output)
        nn = BatchNormalization()(nn)
        nn = PReLU(shared_axes=[1, 2])(nn)
        nn = DepthwiseConv2D(int(nn.shape[1]), depth_multiplier=1, use_bias=False)(nn)
        nn = BatchNormalization()(nn)
        nn = Conv2D(self.emd_size, 1, use_bias=False, activation=None)(nn)

        if 0 < dropout < 1:
            nn = Dropout(dropout)(nn)
        nn = Flatten()(nn)
        nn = Dense(self.emd_size, activation=None, use_bias=False, kernel_initializer="glorot_normal")(nn)
        # embedding = BatchNormalization(name="embedding")(nn)
        norm_emb = Lambda(tf.compat.v1.nn.l2_normalize, name='norm_embedding', arguments={'axis': 1})(nn)
        # basic_model = Model(nw_input, embedding, name=xx.name)
        basic_model = Model(nw_input, norm_emb, name=xx.name)
        return basic_model

    def build_models_v3(self, dropout=0.3):
        seq_layers = [self.__get_bottleneck()]
        base_model = Sequential(seq_layers)
        return base_model


class model_choice(enum.Enum):
    multi_headed_triplet_fn = 1
    single_headed_triplet_fn = 2


def multi_headed_triplet_fn(input_shape=(96, 96, 3), emd_size=128, **kwargs):
    """[summary]

    :param input_shape: [description], defaults to (96, 96, 3)
    :type input_shape: [type], optional
    :param emd_size: [description], defaults to 128
    :type emd_size: [type], optional
    :return: [description]
    :rtype: [type]
    """
    assert len(input_shape) == 3, "Invalid input shape"
    base_model = bottleneck_network(
        input_shape=input_shape,
        emd_size=emd_size, **kwargs
    )(default_model_ver=kwargs.get('version', 'v1'))
    anchor_input = KL.Input(input_shape, name='anchor_input')
    positive_input = KL.Input(input_shape, name='positive_input')
    negative_input = KL.Input(input_shape, name='negative_input')

    # because we re-use the same instance `base_network`, the weights of the network
    # will be shared across the two branches
    encoded_anchor = base_model(anchor_input)
    encoded_positive = base_model(positive_input)
    encoded_negative = base_model(negative_input)
    adam_optim = KO.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)

    merged_vector = KL.concatenate([encoded_anchor, encoded_positive, encoded_negative], axis=-1, name='merged_layer')
    model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=merged_vector)
    model.compile(loss=triplet_loss, optimizer=adam_optim)
    return model, base_model


def single_headed_triplet_fn(input_shape=(96, 96, 3), emd_size=128, **kwargs):
    """[summary]

    :param input_shape: [description], defaults to (96, 96, 3)
    :type input_shape: [type], optional
    :param emd_size: [description], defaults to 128
    :type emd_size: [type], optional
    :return: [description]
    :rtype: [type]
    """
    import tensorflow_addons as tfa
    # from ..common.losses import triplet_loss_adapted_from_tf
    assert len(input_shape) == 3, "Invalid input shape"
    base_model = bottleneck_network(
        input_shape=input_shape,
        emd_size=emd_size, **kwargs
    )(default_model_ver=kwargs.get('version', 'v1'))
    input_images = Input(shape=input_shape)  # input layer for images
    embeddings = base_model([input_images])

    # Defining a model with inputs (images, labels) and outputs (labels_plus_embeddings)
    facemodel = Model(inputs=input_images, outputs=embeddings)
    if kwargs.get('loss_fn', None) == 'semihard':
        facemodel.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=tfa.losses.TripletSemiHardLoss())
    else:
        facemodel.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=tfa.losses.TripletHardLoss())
    return facemodel, base_model


def buildin_models(mode: model_choice, input_shape: typing.Tuple[int] = (96, 96, 3), emd_size: int = 128, **kwargs):
    import sys
    assert isinstance(mode, model_choice), "Invalid model name"
    func = getattr(sys.modules[__name__], mode.name)
    return func(input_shape, emd_size, **kwargs)
