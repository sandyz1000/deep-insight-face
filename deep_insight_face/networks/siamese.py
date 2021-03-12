import numpy as np
from keras.models import Model, Sequential
import keras.layers as KL
import keras.optimizers as KO
from keras import backend as K
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.applications import ResNet50V2, MobileNetV2
# from keras.preprocessing.image import img_to_array


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    y_true = K.cast(y_true, tf.float32)
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def _accuracy(y_true, y_pred, threshold=0.4):
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

    def __init__(self, net="resnet", emd_size=128, input_shape=(112, 112, 3), use_keras_bn=False, **kwargs):
        assert net in ('mobilenet', 'resnet', 'vgg16'), "Invalid bottleneck network"
        self.net = net
        self.emd_size = emd_size
        self.input_shape = input_shape
        self.use_keras_bn = use_keras_bn
        self.kwargs = kwargs

    def __call__(self, dropout=1):
        if self.net == "vgg16":
            return self._vgg_bottleneck(dropout=dropout)
        elif self.use_keras_bn or self.net in ('mobilenet', 'resnet'):
            return self._kbottleneck(dropout=dropout)
        else:
            raise ValueError("Unable to invoke set use_keras_bn=True during initilization.")
            
    def _kbottleneck(self, dropout=0.3):
        _layers = []

        if self.net == "mobilenet":
            _layers.append(MobileNetV2(input_shape=self.input_shape, include_top=False, weights='imagenet'))
        elif self.net == "resnet":
            _layers.append(ResNet50V2(input_shape=self.input_shape, include_top=False, weights='imagenet'))
        else:
            raise AttributeError("Invalid bottleneck network: ", self.net)

        _layers += [
            KL.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'),
            KL.MaxPooling2D(pool_size=2),
            KL.Dropout(0.3),
            KL.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'),
            KL.MaxPooling2D(pool_size=2),
            KL.Dropout(0.3),
            KL.Flatten(name='flatten'),
            KL.Dense(self.emd_size, activation=None, name="embeddings")
        ]
        base_model = Sequential(_layers)
        # base_model.layers[0].trainable = False
        return base_model

    def _vgg_bottleneck(self, dropout=1.0):
        from keras.regularizers import l2
        vgg16_model = VGG16(include_top=False, weights='imagenet', input_shape=self.input_shape)
        
        # vgg16_model.trainable = False
        # for layer in vgg16_model.layers[-4:]:
        #     layer.trainable = True
        out = vgg16_model.output

        out = KL.Conv2D(128, (1, 1), activation='relu', kernel_initializer=initialize_weights,
                        bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4))(out)
        out = KL.MaxPooling2D(padding='same')(out)
        out = KL.Conv2D(128, (1, 1), activation='relu', kernel_initializer=initialize_weights,
                        bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4))(out)
        out = KL.MaxPooling2D(padding='same')(out)

        x = KL.BatchNormalization(name="bn")(out)
        x = KL.Flatten()(out)
        x = KL.Dense(self.emd_size, activation='relu')(x)
        if 0 < dropout < 1:
            x = KL.Dropout(dropout)(x)
        x = KL.Dense(self.emd_size, activation='relu', name="norm_embedding")(x)
        return Model(vgg16_model.input, x)


def buildin_models(emd_size=128, input_shape=(112, 112, 3), summary=False, **kwargs):
    """
    --------------------------------
    Siamese Face Net
    --------------------------------
    """
    assert len(input_shape) == 3, "Invalid input shape"
    # network definition
    base_model = bottleneck_network(emd_size, input_shape, **kwargs)()
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


class SiameseConfig(object):
    model = None
    vgg16_include_top = False
    labels = None
    config = None
    input_shape = None
    threshold = 0.5
    vgg16_model = None
