import os
import enum
import numpy as np
import typing
from numpy import genfromtxt
import tensorflow as tf
from keras.applications import MobileNetV2, ResNet50V2
from keras import optimizers as KO
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, \
    concatenate, DepthwiseConv2D, Dropout, PReLU
import keras.layers as KL
from keras.models import Model, Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.core import Lambda, Flatten, Dense
from keras import backend as K


class faceRecogModel:
    def __init__(self, input_shape=(96, 96, 3), emd_size=128, weights=None) -> None:
        self.input_shape = tuple(input_shape)
        assert self.input_shape == (96, 96, 3), "Invalid Input shape, Shape should be of dimension (96, 96, 3)"
        self.emd_size = emd_size
        self.model = self._load_model()
        if weights is not None:
            assert weights and weights.endswith(".h5"), "Invalid pretrained weights"
            self._load_weights(weights)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __getattr__(self, item):
        return (getattr(self.model, item) if hasattr(self.model, item)
                else getattr(self, item, None))

    def _load_weights(self, model_dir_path):
        """ Load Model weight from csv
        """
        if os.path.basename(model_dir_path).endswith(".h5"):
            return self.model.load_weights(model_dir_path)
        return load_weights_from_FaceNet(self.model, model_dir_path)

    def save_weights(self, model_dir_path):
        assert model_dir_path and model_dir_path.endswith(".h5"), "Invalid weights format"
        self.model.save_weights(model_dir_path)

    def predict_on_batch(self, img):
        return self.model.predict_on_batch(img)

    def _load_model(self):
        myInput = Input(shape=self.input_shape)
        LRN2D = (lambda x: tf.nn.lrn(x, alpha=1e-4, beta=0.75))
        x = ZeroPadding2D(padding=(3, 3), input_shape=self.input_shape)(myInput)
        x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
        x = BatchNormalization(axis=3, epsilon=0.00001, name='bn1')(x)
        x = Activation('relu')(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = MaxPooling2D(pool_size=3, strides=2)(x)
        x = Lambda(LRN2D, name='lrn_1')(x)
        x = Conv2D(64, (1, 1), name='conv2')(x)
        x = BatchNormalization(axis=3, epsilon=0.00001, name='bn2')(x)
        x = Activation('relu')(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(192, (3, 3), name='conv3')(x)
        x = BatchNormalization(axis=3, epsilon=0.00001, name='bn3')(x)
        x = Activation('relu')(x)
        x = Lambda(LRN2D, name='lrn_2')(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = MaxPooling2D(pool_size=3, strides=2)(x)

        # Inception3a
        inception_3a_3x3 = Conv2D(96, (1, 1), name='inception_3a_3x3_conv1')(x)
        inception_3a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_3x3_bn1')(inception_3a_3x3)
        inception_3a_3x3 = Activation('relu')(inception_3a_3x3)
        inception_3a_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3a_3x3)
        inception_3a_3x3 = Conv2D(128, (3, 3), name='inception_3a_3x3_conv2')(inception_3a_3x3)
        inception_3a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_3x3_bn2')(inception_3a_3x3)
        inception_3a_3x3 = Activation('relu')(inception_3a_3x3)

        inception_3a_5x5 = Conv2D(16, (1, 1), name='inception_3a_5x5_conv1')(x)
        inception_3a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_5x5_bn1')(inception_3a_5x5)
        inception_3a_5x5 = Activation('relu')(inception_3a_5x5)
        inception_3a_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3a_5x5)
        inception_3a_5x5 = Conv2D(32, (5, 5), name='inception_3a_5x5_conv2')(inception_3a_5x5)
        inception_3a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_5x5_bn2')(inception_3a_5x5)
        inception_3a_5x5 = Activation('relu')(inception_3a_5x5)

        inception_3a_pool = MaxPooling2D(pool_size=3, strides=2)(x)
        inception_3a_pool = Conv2D(32, (1, 1), name='inception_3a_pool_conv')(inception_3a_pool)
        inception_3a_pool = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_pool_bn')(inception_3a_pool)
        inception_3a_pool = Activation('relu')(inception_3a_pool)
        inception_3a_pool = ZeroPadding2D(padding=((3, 4), (3, 4)))(inception_3a_pool)

        inception_3a_1x1 = Conv2D(64, (1, 1), name='inception_3a_1x1_conv')(x)
        inception_3a_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_1x1_bn')(inception_3a_1x1)
        inception_3a_1x1 = Activation('relu')(inception_3a_1x1)

        inception_3a = concatenate([inception_3a_3x3, inception_3a_5x5, inception_3a_pool, inception_3a_1x1], axis=3)

        # Inception3b
        inception_3b_3x3 = Conv2D(96, (1, 1), name='inception_3b_3x3_conv1')(inception_3a)
        inception_3b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_3x3_bn1')(inception_3b_3x3)
        inception_3b_3x3 = Activation('relu')(inception_3b_3x3)
        inception_3b_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3b_3x3)
        inception_3b_3x3 = Conv2D(128, (3, 3), name='inception_3b_3x3_conv2')(inception_3b_3x3)
        inception_3b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_3x3_bn2')(inception_3b_3x3)
        inception_3b_3x3 = Activation('relu')(inception_3b_3x3)

        inception_3b_5x5 = Conv2D(32, (1, 1), name='inception_3b_5x5_conv1')(inception_3a)
        inception_3b_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_5x5_bn1')(inception_3b_5x5)
        inception_3b_5x5 = Activation('relu')(inception_3b_5x5)
        inception_3b_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3b_5x5)
        inception_3b_5x5 = Conv2D(64, (5, 5), name='inception_3b_5x5_conv2')(inception_3b_5x5)
        inception_3b_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_5x5_bn2')(inception_3b_5x5)
        inception_3b_5x5 = Activation('relu')(inception_3b_5x5)

        inception_3b_pool = Lambda(lambda x: x**2, name='power2_3b')(inception_3a)
        inception_3b_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_3b_pool)
        inception_3b_pool = Lambda(lambda x: x * 9, name='mult9_3b')(inception_3b_pool)
        inception_3b_pool = Lambda(lambda x: K.sqrt(x), name='sqrt_3b')(inception_3b_pool)
        inception_3b_pool = Conv2D(64, (1, 1), name='inception_3b_pool_conv')(inception_3b_pool)
        inception_3b_pool = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_pool_bn')(inception_3b_pool)
        inception_3b_pool = Activation('relu')(inception_3b_pool)
        inception_3b_pool = ZeroPadding2D(padding=(4, 4))(inception_3b_pool)

        inception_3b_1x1 = Conv2D(64, (1, 1), name='inception_3b_1x1_conv')(inception_3a)
        inception_3b_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_1x1_bn')(inception_3b_1x1)
        inception_3b_1x1 = Activation('relu')(inception_3b_1x1)

        inception_3b = concatenate([inception_3b_3x3, inception_3b_5x5, inception_3b_pool, inception_3b_1x1], axis=3)

        # Inception3c
        inception_3c_3x3 = conv2d_bn(inception_3b,
                                     layer='inception_3c_3x3',
                                     cv1_out=128,
                                     cv1_filter=(1, 1),
                                     cv2_out=256,
                                     cv2_filter=(3, 3),
                                     cv2_strides=(2, 2),
                                     padding=(1, 1))

        inception_3c_5x5 = conv2d_bn(inception_3b,
                                     layer='inception_3c_5x5',
                                     cv1_out=32,
                                     cv1_filter=(1, 1),
                                     cv2_out=64,
                                     cv2_filter=(5, 5),
                                     cv2_strides=(2, 2),
                                     padding=(2, 2))

        inception_3c_pool = MaxPooling2D(pool_size=3, strides=2)(inception_3b)
        inception_3c_pool = ZeroPadding2D(padding=((0, 1), (0, 1)))(inception_3c_pool)

        inception_3c = concatenate([inception_3c_3x3, inception_3c_5x5, inception_3c_pool], axis=3)

        # inception 4a
        inception_4a_3x3 = conv2d_bn(inception_3c,
                                     layer='inception_4a_3x3',
                                     cv1_out=96,
                                     cv1_filter=(1, 1),
                                     cv2_out=192,
                                     cv2_filter=(3, 3),
                                     cv2_strides=(1, 1),
                                     padding=(1, 1))
        inception_4a_5x5 = conv2d_bn(inception_3c,
                                     layer='inception_4a_5x5',
                                     cv1_out=32,
                                     cv1_filter=(1, 1),
                                     cv2_out=64,
                                     cv2_filter=(5, 5),
                                     cv2_strides=(1, 1),
                                     padding=(2, 2))

        inception_4a_pool = Lambda(lambda x: x**2, name='power2_4a')(inception_3c)
        inception_4a_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_4a_pool)
        inception_4a_pool = Lambda(lambda x: x * 9, name='mult9_4a')(inception_4a_pool)
        inception_4a_pool = Lambda(lambda x: K.sqrt(x), name='sqrt_4a')(inception_4a_pool)
        inception_4a_pool = conv2d_bn(inception_4a_pool,
                                      layer='inception_4a_pool',
                                      cv1_out=128,
                                      cv1_filter=(1, 1),
                                      padding=(2, 2))
        inception_4a_1x1 = conv2d_bn(inception_3c,
                                     layer='inception_4a_1x1',
                                     cv1_out=256,
                                     cv1_filter=(1, 1))
        inception_4a = concatenate([inception_4a_3x3, inception_4a_5x5, inception_4a_pool, inception_4a_1x1], axis=3)

        # inception4e
        inception_4e_3x3 = conv2d_bn(inception_4a,
                                     layer='inception_4e_3x3',
                                     cv1_out=160,
                                     cv1_filter=(1, 1),
                                     cv2_out=256,
                                     cv2_filter=(3, 3),
                                     cv2_strides=(2, 2),
                                     padding=(1, 1))
        inception_4e_5x5 = conv2d_bn(inception_4a,
                                     layer='inception_4e_5x5',
                                     cv1_out=64,
                                     cv1_filter=(1, 1),
                                     cv2_out=128,
                                     cv2_filter=(5, 5),
                                     cv2_strides=(2, 2),
                                     padding=(2, 2))
        inception_4e_pool = MaxPooling2D(pool_size=3, strides=2)(inception_4a)
        inception_4e_pool = ZeroPadding2D(padding=((0, 1), (0, 1)))(inception_4e_pool)

        inception_4e = concatenate([inception_4e_3x3, inception_4e_5x5, inception_4e_pool], axis=3)

        # inception5a
        inception_5a_3x3 = conv2d_bn(inception_4e,
                                     layer='inception_5a_3x3',
                                     cv1_out=96,
                                     cv1_filter=(1, 1),
                                     cv2_out=384,
                                     cv2_filter=(3, 3),
                                     cv2_strides=(1, 1),
                                     padding=(1, 1))

        inception_5a_pool = Lambda(lambda x: x**2, name='power2_5a')(inception_4e)
        inception_5a_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_5a_pool)
        inception_5a_pool = Lambda(lambda x: x * 9, name='mult9_5a')(inception_5a_pool)
        inception_5a_pool = Lambda(lambda x: K.sqrt(x), name='sqrt_5a')(inception_5a_pool)
        inception_5a_pool = conv2d_bn(inception_5a_pool,
                                      layer='inception_5a_pool',
                                      cv1_out=96,
                                      cv1_filter=(1, 1),
                                      padding=(1, 1))
        inception_5a_1x1 = conv2d_bn(inception_4e,
                                     layer='inception_5a_1x1',
                                     cv1_out=256,
                                     cv1_filter=(1, 1))

        inception_5a = concatenate([inception_5a_3x3, inception_5a_pool, inception_5a_1x1], axis=3)

        # inception_5b
        inception_5b_3x3 = conv2d_bn(inception_5a,
                                     layer='inception_5b_3x3',
                                     cv1_out=96,
                                     cv1_filter=(1, 1),
                                     cv2_out=384,
                                     cv2_filter=(3, 3),
                                     cv2_strides=(1, 1),
                                     padding=(1, 1))
        inception_5b_pool = MaxPooling2D(pool_size=3, strides=2)(inception_5a)
        inception_5b_pool = conv2d_bn(inception_5b_pool,
                                      layer='inception_5b_pool',
                                      cv1_out=96,
                                      cv1_filter=(1, 1))
        inception_5b_pool = ZeroPadding2D(padding=(1, 1))(inception_5b_pool)

        inception_5b_1x1 = conv2d_bn(inception_5a,
                                     layer='inception_5b_1x1',
                                     cv1_out=256,
                                     cv1_filter=(1, 1))
        inception_5b = concatenate([inception_5b_3x3, inception_5b_pool, inception_5b_1x1], axis=3)

        av_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1))(inception_5b)
        reshape_layer = Flatten()(av_pool)
        dense_layer = Dense(self.emd_size, name='dense_layer')(reshape_layer)
        norm_layer = Lambda(lambda x: K.l2_normalize(x, axis=1), name='norm_layer')(dense_layer)

        # Final Model
        model = Model(inputs=[myInput], outputs=norm_layer)
        return model


def conv2d_bn(
    x,
    layer=None,
    cv1_out=None,
    cv1_filter=(1, 1),
    cv1_strides=(1, 1),
    cv2_out=None,
    cv2_filter=(3, 3),
    cv2_strides=(1, 1),
    padding=None,
):
    num = '' if cv2_out is None else '1'
    tensor = Conv2D(cv1_out, cv1_filter, strides=cv1_strides, name=layer + '_conv' + num)(x)
    tensor = BatchNormalization(axis=3, epsilon=0.00001, name=layer + '_bn' + num)(tensor)
    tensor = Activation('relu')(tensor)
    if padding is None:
        return tensor
    tensor = ZeroPadding2D(padding=padding)(tensor)
    if cv2_out is None:
        return tensor
    tensor = Conv2D(cv2_out, cv2_filter, strides=cv2_strides, name=layer + '_conv' + '2')(tensor)
    tensor = BatchNormalization(axis=3, epsilon=0.00001, name=layer + '_bn' + '2')(tensor)
    tensor = Activation('relu')(tensor)
    return tensor


WEIGHTS = [
    'conv1', 'bn1', 'conv2', 'bn2', 'conv3', 'bn3',
    'inception_3a_1x1_conv', 'inception_3a_1x1_bn',
    'inception_3a_pool_conv', 'inception_3a_pool_bn',
    'inception_3a_5x5_conv1', 'inception_3a_5x5_conv2', 'inception_3a_5x5_bn1', 'inception_3a_5x5_bn2',
    'inception_3a_3x3_conv1', 'inception_3a_3x3_conv2', 'inception_3a_3x3_bn1', 'inception_3a_3x3_bn2',
    'inception_3b_3x3_conv1', 'inception_3b_3x3_conv2', 'inception_3b_3x3_bn1', 'inception_3b_3x3_bn2',
    'inception_3b_5x5_conv1', 'inception_3b_5x5_conv2', 'inception_3b_5x5_bn1', 'inception_3b_5x5_bn2',
    'inception_3b_pool_conv', 'inception_3b_pool_bn',
    'inception_3b_1x1_conv', 'inception_3b_1x1_bn',
    'inception_3c_3x3_conv1', 'inception_3c_3x3_conv2', 'inception_3c_3x3_bn1', 'inception_3c_3x3_bn2',
    'inception_3c_5x5_conv1', 'inception_3c_5x5_conv2', 'inception_3c_5x5_bn1', 'inception_3c_5x5_bn2',
    'inception_4a_3x3_conv1', 'inception_4a_3x3_conv2', 'inception_4a_3x3_bn1', 'inception_4a_3x3_bn2',
    'inception_4a_5x5_conv1', 'inception_4a_5x5_conv2', 'inception_4a_5x5_bn1', 'inception_4a_5x5_bn2',
    'inception_4a_pool_conv', 'inception_4a_pool_bn',
    'inception_4a_1x1_conv', 'inception_4a_1x1_bn',
    'inception_4e_3x3_conv1', 'inception_4e_3x3_conv2', 'inception_4e_3x3_bn1', 'inception_4e_3x3_bn2',
    'inception_4e_5x5_conv1', 'inception_4e_5x5_conv2', 'inception_4e_5x5_bn1', 'inception_4e_5x5_bn2',
    'inception_5a_3x3_conv1', 'inception_5a_3x3_conv2', 'inception_5a_3x3_bn1', 'inception_5a_3x3_bn2',
    'inception_5a_pool_conv', 'inception_5a_pool_bn',
    'inception_5a_1x1_conv', 'inception_5a_1x1_bn',
    'inception_5b_3x3_conv1', 'inception_5b_3x3_conv2', 'inception_5b_3x3_bn1', 'inception_5b_3x3_bn2',
    'inception_5b_pool_conv', 'inception_5b_pool_bn',
    'inception_5b_1x1_conv', 'inception_5b_1x1_bn',
    'dense_layer'
]

conv_shape = {
    'conv1': [64, 3, 7, 7],
    'conv2': [64, 64, 1, 1],
    'conv3': [192, 64, 3, 3],
    'inception_3a_1x1_conv': [64, 192, 1, 1],
    'inception_3a_pool_conv': [32, 192, 1, 1],
    'inception_3a_5x5_conv1': [16, 192, 1, 1],
    'inception_3a_5x5_conv2': [32, 16, 5, 5],
    'inception_3a_3x3_conv1': [96, 192, 1, 1],
    'inception_3a_3x3_conv2': [128, 96, 3, 3],
    'inception_3b_3x3_conv1': [96, 256, 1, 1],
    'inception_3b_3x3_conv2': [128, 96, 3, 3],
    'inception_3b_5x5_conv1': [32, 256, 1, 1],
    'inception_3b_5x5_conv2': [64, 32, 5, 5],
    'inception_3b_pool_conv': [64, 256, 1, 1],
    'inception_3b_1x1_conv': [64, 256, 1, 1],
    'inception_3c_3x3_conv1': [128, 320, 1, 1],
    'inception_3c_3x3_conv2': [256, 128, 3, 3],
    'inception_3c_5x5_conv1': [32, 320, 1, 1],
    'inception_3c_5x5_conv2': [64, 32, 5, 5],
    'inception_4a_3x3_conv1': [96, 640, 1, 1],
    'inception_4a_3x3_conv2': [192, 96, 3, 3],
    'inception_4a_5x5_conv1': [32, 640, 1, 1, ],
    'inception_4a_5x5_conv2': [64, 32, 5, 5],
    'inception_4a_pool_conv': [128, 640, 1, 1],
    'inception_4a_1x1_conv': [256, 640, 1, 1],
    'inception_4e_3x3_conv1': [160, 640, 1, 1],
    'inception_4e_3x3_conv2': [256, 160, 3, 3],
    'inception_4e_5x5_conv1': [64, 640, 1, 1],
    'inception_4e_5x5_conv2': [128, 64, 5, 5],
    'inception_5a_3x3_conv1': [96, 1024, 1, 1],
    'inception_5a_3x3_conv2': [384, 96, 3, 3],
    'inception_5a_pool_conv': [96, 1024, 1, 1],
    'inception_5a_1x1_conv': [256, 1024, 1, 1],
    'inception_5b_3x3_conv1': [96, 736, 1, 1],
    'inception_5b_3x3_conv2': [384, 96, 3, 3],
    'inception_5b_pool_conv': [96, 736, 1, 1],
    'inception_5b_1x1_conv': [256, 736, 1, 1],
}


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


def load_weights_from_FaceNet(FRmodel, model_dir_path=None):
    # Load weights from csv files (which was exported from Openface torch model)
    weights = WEIGHTS
    weights_dict = load_weights(model_dir_path)

    # Set layer weights of the model
    for name in weights:
        if FRmodel.get_layer(name) is not None:
            FRmodel.get_layer(name).set_weights(weights_dict[name])
        elif FRmodel.get_layer(name) is not None:
            FRmodel.get_layer(name).set_weights(weights_dict[name])


def load_weights(model_dir_path="./facenet-weights"):
    assert model_dir_path is not None, "Invalid model directory path"

    # Set weights path
    dirPath = model_dir_path
    fileNames = filter(lambda f: not f.startswith('.'), os.listdir(dirPath))
    paths = {}
    weights_dict = {}

    for n in fileNames:
        paths[n.replace('.csv', '')] = dirPath + os.path.sep + n

    for name in WEIGHTS:
        if 'conv' in name:
            conv_w = genfromtxt(paths[name + '_w'], delimiter=',', dtype=None)
            conv_w = np.reshape(conv_w, conv_shape[name])
            conv_w = np.transpose(conv_w, (2, 3, 1, 0))
            conv_b = genfromtxt(paths[name + '_b'], delimiter=',', dtype=None)
            weights_dict[name] = [conv_w, conv_b]
        elif 'bn' in name:
            bn_w = genfromtxt(paths[name + '_w'], delimiter=',', dtype=None)
            bn_b = genfromtxt(paths[name + '_b'], delimiter=',', dtype=None)
            bn_m = genfromtxt(paths[name + '_m'], delimiter=',', dtype=None)
            bn_v = genfromtxt(paths[name + '_v'], delimiter=',', dtype=None)
            weights_dict[name] = [bn_w, bn_b, bn_m, bn_v]
        elif 'dense' in name:
            dense_w = genfromtxt(dirPath + '/dense_w.csv', delimiter=',', dtype=None)
            dense_w = np.reshape(dense_w, (128, 736))
            dense_w = np.transpose(dense_w, (1, 0))
            dense_b = genfromtxt(dirPath + '/dense_b.csv', delimiter=',', dtype=None)
            weights_dict[name] = [dense_w, dense_b]

    return weights_dict


class bottleneck_network:
    """
    ------------------------------
    FaceNet Triplet configuration
    ------------------------------
    Return faceRecognition model based on inceptionv2 network

    Keyword Arguments:
        model_dir_path {[type]} -- [description] (default: {None})
        emd_size {[type]} -- [description] (default: {128})
        input_shape {[type]} -- [description] (default: {(96, 96, 3)})
    """

    def __init__(self, net="resnet",
                 emd_size=128,
                 input_shape=(96, 96, 3), use_keras_bn=False, **kwargs):
        assert net in ('mobilenet', 'resnet', 'facereco'), "Invalid bottleneck network"
        self.net = net
        self.emd_size = emd_size
        self.input_shape = input_shape
        self.use_keras_bn = use_keras_bn
        self.kwargs = kwargs

    def __call__(self, dropout=1) -> typing.Any:
        if self.net == 'facereco':
            return self._face_reco_model()
        elif self.use_keras_bn or self.net in ('mobilenet', 'resnet'):
            return self._kbottleneck_v1(dropout=dropout)
        else:
            raise ValueError("Unable to invoke set use_keras_bn=True during initilization.")

    def _kbottleneck_v1(self, dropout=0.3):
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
            KL.Dense(self.emd_size, activation=None, name="embeddings"),  # No activation on final dense layer
            # KL.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))  # L2 normalize embeddings
        ]
        base_model = Sequential(_layers)
        # base_model.layers[0].trainable = False
        return base_model

    def _kbottleneck(self, dropout=1):
        if self.net == 'mobilenet':
            xx = MobileNetV2(input_shape=self.input_shape, include_top=False, weights='imagenet')

        elif self.net == 'resnet':
            xx = ResNet50V2(input_shape=self.input_shape, include_top=False, weights="imagenet")

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

    def _face_reco_model(self):
        inception_model = faceRecogModel(input_shape=self.input_shape, emd_size=self.emd_size,
                                         weights=self.kwargs.get('weights', None))
        print("Total Params:", inception_model.count_params())
        return inception_model


class model_choice(enum.Enum):
    choice_1 = "simple_triplet_nw"
    choice_2 = "semihard_triplet_nw"


def simple_triplet_nw(input_shape=(96, 96, 3), emd_size=128, summary=True, **kwargs):
    assert len(input_shape) == 3, "Invalid input shape"
    # network definition
    base_model = bottleneck_network(input_shape=input_shape, emd_size=emd_size, **kwargs)()
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

    print(model.summary() if summary else ">>>>> SIMPLE_TRIPLET_NW Model sucessfully loaded >>>>> ")
    return model, base_model


def semihard_triplet_nw(input_shape=(96, 96, 3), emd_size=128, summary=True, **kwargs):
    import tensorflow_addons as tfa
    # from ..handlers.losses import triplet_loss_adapted_from_tf
    assert len(input_shape) == 3, "Invalid input shape"
    base_model = bottleneck_network(input_shape=input_shape, emd_size=emd_size, **kwargs)()
    input_images = Input(shape=input_shape)  # input layer for images
    embeddings = base_model([input_images])

    # Defining a model with inputs (images, labels) and outputs (labels_plus_embeddings)
    facemodel = Model(inputs=input_images, outputs=embeddings)
    facemodel.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=tfa.losses.TripletSemiHardLoss())
    # facemodel.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=triplet_loss_adapted_from_tf)

    print(facemodel.summary() if summary else ">>>>> SEMIHARD_TRIPLET_NW Model sucessfully loaded >>>>> ")
    return facemodel, base_model


def buildin_models(mode: model_choice, input_shape: typing.Tuple[int] = (96, 96, 3), emd_size: int = 128, **kwargs):
    import sys
    assert isinstance(mode, model_choice), "Invalid model name"
    func = getattr(sys.modules[__name__], mode.name)
    return func(input_shape, emd_size, **kwargs)
