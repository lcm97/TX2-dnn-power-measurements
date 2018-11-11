from keras.models import Model
from keras.layers import Conv2D, Input, Flatten, Dense, MaxPooling1D, TimeDistributed, Lambda, Reshape
from keras.layers.merge import concatenate
from Data import *
from keras import backend as K
import tensorflow as tf


def lambda_fun(x, num_split, index):
    split = tf.split(x, num_split, axis=1)
    split_index = split[index]
    return split_index


def spatial_model():
    image = Input(shape=(12, 16, 3))

    layer = Conv2D(256, (5, 5))(image)
    layer = Conv2D(256, (3, 3))(layer)
    layer = Conv2D(256, (3, 3))(layer)
    layer = Flatten()(layer)

    layer = Dense(256, activation='relu')(layer)

    return Model(image, layer)


def temporal_model():
    image = Input(shape=(12, 16, 20))

    layer = Conv2D(256, (5, 5))(image)
    layer = Conv2D(256, (3, 3))(layer)
    layer = Conv2D(256, (3, 3))(layer)
    layer = Flatten()(layer)
    layer = Dense(256, activation='relu')(layer)

    return Model(image, layer)


def maxpoolings():
    inter_represent = Input(shape=(16, 256))
    """level 1"""
    layer1 = MaxPooling1D(pool_size=1, strides=256, padding="SAME")(inter_represent)
    """level 2"""
    sub_layer1 = Lambda(lambda_fun, arguments={'num_split': 2, 'index': 0})(inter_represent)
    sub_layer2 = Lambda(lambda_fun, arguments={'num_split': 2, 'index': 1})(inter_represent)
    layer2 = MaxPooling1D(pool_size=1, strides=256, padding="SAME")(sub_layer1)
    layer3 = MaxPooling1D(pool_size=1, strides=256, padding="SAME")(sub_layer2)
    """level 3"""
    sub_layer3 = Lambda(lambda_fun, arguments={'num_split': 4, 'index': 0})(inter_represent)
    sub_layer4 = Lambda(lambda_fun, arguments={'num_split': 4, 'index': 1})(inter_represent)
    sub_layer5 = Lambda(lambda_fun, arguments={'num_split': 4, 'index': 2})(inter_represent)
    sub_layer6 = Lambda(lambda_fun, arguments={'num_split': 4, 'index': 3})(inter_represent)
    layer4 = MaxPooling1D(pool_size=1, strides=256, padding="SAME")(sub_layer3)
    layer5 = MaxPooling1D(pool_size=1, strides=256, padding="SAME")(sub_layer4)
    layer6 = MaxPooling1D(pool_size=1, strides=256, padding="SAME")(sub_layer5)
    layer7 = MaxPooling1D(pool_size=1, strides=256, padding="SAME")(sub_layer6)
    """level 4"""
    sub_layer7 = Lambda(lambda_fun, arguments={'num_split': 8, 'index': 0})(inter_represent)
    sub_layer8 = Lambda(lambda_fun, arguments={'num_split': 8, 'index': 1})(inter_represent)
    sub_layer9 = Lambda(lambda_fun, arguments={'num_split': 8, 'index': 2})(inter_represent)
    sub_layer10 = Lambda(lambda_fun, arguments={'num_split': 8, 'index': 3})(inter_represent)
    sub_layer11 = Lambda(lambda_fun, arguments={'num_split': 8, 'index': 4})(inter_represent)
    sub_layer12 = Lambda(lambda_fun, arguments={'num_split': 8, 'index': 5})(inter_represent)
    sub_layer13 = Lambda(lambda_fun, arguments={'num_split': 8, 'index': 6})(inter_represent)
    sub_layer14 = Lambda(lambda_fun, arguments={'num_split': 8, 'index': 7})(inter_represent)
    layer8 = MaxPooling1D(pool_size=1, strides=256, padding="SAME")(sub_layer7)
    layer9 = MaxPooling1D(pool_size=1, strides=256, padding="SAME")(sub_layer8)
    layer10 = MaxPooling1D(pool_size=1, strides=256, padding="SAME")(sub_layer9)
    layer11 = MaxPooling1D(pool_size=1, strides=256, padding="SAME")(sub_layer10)
    layer12 = MaxPooling1D(pool_size=1, strides=256, padding="SAME")(sub_layer11)
    layer13 = MaxPooling1D(pool_size=1, strides=256, padding="SAME")(sub_layer12)
    layer14 = MaxPooling1D(pool_size=1, strides=256, padding="SAME")(sub_layer13)
    layer15 = MaxPooling1D(pool_size=1, strides=256, padding="SAME")(sub_layer14)
    layer = concatenate([layer1, layer2, layer3, layer4, layer5, layer6, layer7, layer8,
                         layer9, layer10, layer11, layer12, layer13, layer14, layer15], 1)
    """the out put shape is (15,256),for each maxpooling layer generate (1,256)"""
    return Model(inter_represent, layer)


def spatial_model_multi():
    """Input with 16x(12,16,3) frames with time distribute"""
    frames = Input(shape=(16, 12, 16, 3))
    layer = TimeDistributed(spatial_model())(frames)
    """The output shape is (16,256)"""
    return Model(frames, layer)


def temporal_model_multi():
    optical_flows = Input(shape=(16, 12, 16, 20))
    layer = TimeDistributed(temporal_model())(optical_flows)
    return Model(optical_flows, layer)


def temporal_pyramid_concate():
    spatial_input = Input(shape=(1, 15, 256))
    temporal_input = Input(shape=(1, 15, 256))

    """Now,the output shape is (2,15,256)"""
    layer = concatenate([spatial_input, temporal_input], 1)
    return Model([spatial_input, temporal_input], layer)


def final_dense_layers():
    layer_input = Input(shape=(2, 15, 256))
    layer = Flatten()(layer_input)
    layer = Dense(8192, activation='relu')(layer)
    layer = Dense(8192, activation='relu')(layer)
    layer = Dense(51, activation='softmax')(layer)
    return Model(layer_input, layer)


def two_stream_model():
    frames = Input(shape=(16, 12, 16, 3))
    optical_flows = Input(shape=(16, 12, 16, 20))

    """Spatial stream and Temporal stream"""
    spatial_result = spatial_model_multi()(frames)
    tempo_result = temporal_model_multi()(optical_flows)
    """Maxpoolings """
    spatial_result = maxpoolings()(spatial_result)
    tempo_result = maxpoolings()(tempo_result)
    """Two stream concat"""
    pyramid_concat = temporal_pyramid_concate()([spatial_result, tempo_result])
    """Final dense layer"""
    dense_result = final_dense_layers()(pyramid_concat)

    return Model([frames, optical_flows], dense_result)

