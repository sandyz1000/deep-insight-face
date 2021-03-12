from tensorflow import keras
import tensorflow.keras.backend as K
import tensorflow as tf
from typing import Tuple
from tensorflow.keras.applications import (ResNet101, ResNet101V2, MobileNet,
                                           MobileNetV2, NASNetMobile, ResNet50, ResNet50V2)


def print_buildin_models():
    print("""
    >>>> buildin_models
    mobilenet, mobilenetv2, mobilenetv3_small, mobilenetv3_large, mobilefacenet, se_mobilefacenet,
    nasnetmobile resnet50v2, resnet101v2, se_resnext, resnest50, resnest101,
    efficientnetb0, efficientnetb1, efficientnetb2, efficientnetb3, efficientnetb4, efficientnetb5,
    efficientnetb6, efficientnetb7,
    """, end='')


def buildin_models(
    name: str, dropout: int = 1, emb_shape: int = 512, input_shape: Tuple[int] = (112, 112, 3),
    output_layer: str = "GDC", bn_momentum: float = 0.99, bn_epsilon: float = 0.001, **kwargs
):
    name = name.lower()
    # Basic Model
    if name == "mobilenet":
        xx = MobileNet(input_shape=input_shape, include_top=False, weights=None, **kwargs)
    elif name == "mobilenetv2":
        xx = MobileNetV2(input_shape=input_shape, include_top=False, weights=None, **kwargs)
    elif name == "resnet34":
        from ..backbones import resnet

        xx = resnet.ResNet34(input_shape=input_shape, include_top=False, weights=None, **kwargs)
    elif name == "resnet50":
        xx = ResNet50(input_shape=input_shape, include_top=False, weights="imagenet", **kwargs)
    elif name == "resnet50v2":
        xx = ResNet50V2(input_shape=input_shape, include_top=False, weights="imagenet", **kwargs)
    elif name == "resnet101":
        xx = ResNet101(input_shape=input_shape, include_top=False, weights=None, **kwargs)
    elif name == "resnet101v2":
        xx = ResNet101V2(input_shape=input_shape, include_top=False, weights="imagenet", **kwargs)
    elif name == "nasnetmobile":
        xx = NASNetMobile(input_shape=input_shape, include_top=False, weights=None, **kwargs)
    elif name.startswith("efficientnet"):
        if "-dev" in tf.__version__:
            # import tensorflow.keras.applications.efficientnet as efficientnet
            from ..backbones import efficientnet
        else:
            # import efficientnet.tfkeras as efficientnet
            from ..backbones import efficientnet

        if name[-2] == "b":
            compound_scale = int(name[-1])
            models = [
                efficientnet.EfficientNetB0,
                efficientnet.EfficientNetB1,
                efficientnet.EfficientNetB2,
                efficientnet.EfficientNetB3,
                efficientnet.EfficientNetB4,
                efficientnet.EfficientNetB5,
                efficientnet.EfficientNetB6,
                efficientnet.EfficientNetB7,
            ]
            model = models[compound_scale]
        else:
            model = efficientnet.EfficientNetL2
        xx = model(weights="noisy-student", include_top=False, input_shape=input_shape)  # or weights='imagenet'
    elif name.startswith("se_resnext"):
        from keras_squeeze_excite_network import se_resnext

        if name.endswith("101"):  # se_resnext101
            depth = [3, 4, 23, 3]
        else:  # se_resnext50
            depth = [3, 4, 6, 3]
        xx = se_resnext.SEResNextImageNet(weights="imagenet", input_shape=input_shape, include_top=False, depth=depth)
    elif name.startswith("resnest"):
        from ..backbones import resnest

        if name == "resnest50":
            xx = resnest.ResNest50(input_shape=input_shape)
        else:
            xx = resnest.ResNest101(input_shape=input_shape)
    elif name.startswith("mobilenetv3"):
        from ..backbones import mobilenetv3

        size = "small" if "small" in name else "large"
        xx = mobilenetv3.MobilenetV3(input_shape=input_shape, include_top=False, size=size)
    elif "mobilefacenet" in name or "mobile_facenet" in name:
        from ..backbones import mobile_facenet

        use_se = True if "se" in name else False
        xx = mobile_facenet.mobile_facenet(input_shape=input_shape, include_top=False, name=name, use_se=use_se)
    else:
        return None
    # xx = keras.models.load_model('checkpoints/mobilnet_v1_basic_922667.h5', compile=False)
    xx.trainable = False

    inputs = xx.inputs[0]
    nn = xx.outputs[0]
    # nn = keras.layers.Conv2D(emb_shape, xx.output_shape[1], use_bias=False)(nn)

    if output_layer == "E":
        """ Fully Connected """
        nn = keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=bn_epsilon)(nn)
        if dropout > 0 and dropout < 1:
            nn = keras.layers.Dropout(dropout)(nn)
        nn = keras.layers.Flatten()(nn)
        nn = keras.layers.Dense(emb_shape, activation=None, use_bias=True, kernel_initializer="glorot_normal")(nn)
    else:
        """ GDC """
        # nn = keras.layers.Conv2D(512, 1, use_bias=False)(nn)
        # nn = keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=bn_epsilon)(nn)
        # nn = keras.layers.PReLU(shared_axes=[1, 2])(nn)
        nn = keras.layers.DepthwiseConv2D(int(nn.shape[1]), depth_multiplier=1, use_bias=False)(nn)
        nn = keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=bn_epsilon)(nn)
        if dropout > 0 and dropout < 1:
            nn = keras.layers.Dropout(dropout)(nn)
        nn = keras.layers.Conv2D(emb_shape, 1, use_bias=False, activation=None, kernel_initializer="glorot_normal")(nn)
        nn = keras.layers.Flatten()(nn)
        # nn = keras.layers.Dense(emb_shape, activation=None, use_bias=True, kernel_initializer="glorot_normal")(nn)
    embedding = keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=bn_epsilon, name="embedding")(nn)
    basic_model = keras.models.Model(inputs, embedding, name=xx.name)
    return basic_model


class NormDense(keras.layers.Layer):
    def __init__(self, units=1000, **kwargs):
        super(NormDense, self).__init__(**kwargs)
        self.init = keras.initializers.glorot_normal()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            name="norm_dense_w", shape=(input_shape[-1], self.units), initializer=self.init, trainable=True
        )
        super(NormDense, self).build(input_shape)

    def call(self, inputs, **kwargs):
        norm_w = K.l2_normalize(self.w, axis=0)
        inputs = K.l2_normalize(inputs, axis=1)
        return K.dot(inputs, norm_w)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

    def get_config(self):
        config = super(NormDense, self).get_config()
        config.update({"units": self.units})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
