import typing
from dataclasses import dataclass
import numpy as np
from keras.models import Model, Sequential
import keras.layers as KL
import keras.optimizers as KO
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2, MobileNetV2, VGG16


class SiameseConfig(typing.NamedTuple):
    model: tf.keras.Model = None
    vgg16_include_top: bool = False
    labels: str = None
    config: str = None
    input_shape: typing.Tuple = None
    threshold: float = 0.5
    vgg16_model: tf.keras.Model = None


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true: tf.Tensor, y_pred: tf.Tensor):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    y_true = K.cast(y_true, tf.float32)
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def _accuracy(y_true: tf.Tensor, y_pred: tf.Tensor, threshold=0.4):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < threshold, y_true.dtype)))


def initialize_weights(shape, name=None, dtype=None):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer weights with mean as 0.0 and standard deviation of 0.01
    """
    return np.random.normal(loc=0.0, scale=1e-2, size=shape)


def initialize_bias(shape, name=None, dtype=None):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer bias with mean as 0.5 and standard deviation of 0.01
    """
    return np.random.normal(loc=0.5, scale=1e-2, size=shape)


class bottleneck_network:
    '''Base network to be shared (eq. to feature extraction).
    '''

    def __init__(self, net="resnet", emd_size=128, input_shape=(112, 112, 3), **kwargs):
        assert net in ('mobilenet', 'resnet', 'vgg16'), "Invalid bottleneck network"
        self.net = net
        self.emd_size = emd_size
        self.input_shape = input_shape
        self.kwargs = kwargs

    def __call__(self, default_model='v1', dropout=0.2):
        attr_name = 'build_models_' + default_model
        assert hasattr(self, attr_name), "Invalid default model version, must be from options (v1, v2)"
        base_model = getattr(self, attr_name)(dropout=dropout)
        return base_model

    def __get_bottleneck(self, ):
        if self.net == "mobilenet":
            bottleneck = MobileNetV2(input_shape=self.input_shape, include_top=False, weights='imagenet')
        elif self.net == "resnet":
            bottleneck = ResNet50V2(input_shape=self.input_shape, include_top=False, weights='imagenet')
        elif self.net == "vgg16":
            bottleneck = VGG16(include_top=False, weights='imagenet', input_shape=self.input_shape)

        return bottleneck

    def build_models_v1(self, dropout: float = 0.3):
        sequentials = [self.__get_bottleneck()]
        sequentials += [
            KL.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'),
            KL.MaxPooling2D(pool_size=2),
            KL.Dropout(0.3),
            KL.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'),
            KL.MaxPooling2D(pool_size=2),
            KL.Dropout(0.3),
            KL.Flatten(name='flatten'),
            KL.Dense(self.emd_size, activation=None, name="embeddings")
        ]
        base_model = Sequential(sequentials)
        # base_model.layers[0].trainable = False
        return base_model

    def build_models_v2(self, dropout: float = 1.0):
        from keras.regularizers import l2
        layers = [self.__get_bottleneck()]
        layers += [
            KL.Conv2D(128, (1, 1), activation='relu',
                      kernel_initializer=initialize_weights,
                      bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)
                      ),
            KL.MaxPooling2D(padding='same'),
            KL.Conv2D(128, (1, 1), activation='relu',
                      kernel_initializer=initialize_weights,
                      bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)
                      ),
            KL.MaxPooling2D(padding='same'),
            KL.BatchNormalization(name="bn"),
            KL.Flatten(),
            KL.Dropout(dropout),
            KL.Dense(self.emd_size, activation='relu', name="norm_embedding")
        ]
        base_model = Sequential(layers)
        # base_model.layers[0].trainable = False
        return base_model


def buildin_models(
    emd_size: int = 128,
    input_shape: typing.Tuple[int] = (112, 112, 3),
    summary: bool = False, 
    **kwargs
) -> typing.Tuple[tf.keras.Model, tf.keras.Model]:
    """
    --------------------------------
    Siamese Face Net
    --------------------------------
    """
    assert len(input_shape) == 3, "Invalid input shape"
    # network definition
    base_model = bottleneck_network(emd_size, input_shape, **kwargs)(default_model='v1')
    input_a = KL.Input(shape=input_shape)
    input_b = KL.Input(shape=input_shape)

    # because we re-use the same instance `base_network`, the weights of the network
    # will be shared across the two branches
    processed_a = base_model(input_a)
    processed_b = base_model(input_b)

    distance = KL.Lambda(euclidean_distance,
                         output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model = Model([input_a, input_b], distance)
    optimizer = KO.Adam(lr=0.00006)
    model.compile(loss=contrastive_loss, optimizer=optimizer, metrics=[_accuracy])
    # model.compile(loss="binary_crossentropy", optimizer=optimizer)
    print(model.summary() if summary else ">>>>>>> Siamese MODEL Loaded >>>>>>>")
    return model, base_model
