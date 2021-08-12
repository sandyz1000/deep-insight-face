import typing
import os
import numpy as np
from numpy import genfromtxt
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.layers import (
    Conv2D, ZeroPadding2D, Activation, Input,
    MaxPooling2D, AveragePooling2D, BatchNormalization, Lambda,
    Flatten, Dense, concatenate,
)
from keras.models import Model


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


class InceptionNetwork:
    def __init__(self, input_shape: typing.Tuple = (96, 96, 3), emd_size: int = 128, weights=None) -> None:
        self.input_shape = tuple(input_shape)
        assert self.input_shape == (96, 96, 3), "Invalid Input shape, Shape should be of dimension (96, 96, 3)"
        self.emd_size = emd_size
        self.model = self._load_model()
        if weights is not None:
            assert weights and weights.endswith(".h5"), "Invalid pretrained weights"
            self._load_weights(weights)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __getattr__(self, item: str):
        return getattr(self.model, item) if hasattr(self.model, item) else getattr(self, item, None)

    def _load_weights(self, model_dir_path: str):
        """ Load Model weight from csv
        """
        if os.path.basename(model_dir_path).endswith(".h5"):
            return self.model.load_weights(model_dir_path)
        return load_weights_from_FaceNet(self.model, model_dir_path)

    def save_weights(self, model_dir_path: str):
        assert model_dir_path and model_dir_path.endswith(".h5"), "Invalid weights format"
        self.model.save_weights(model_dir_path)

    def predict_on_batch(self, img: np.ndarray):
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
