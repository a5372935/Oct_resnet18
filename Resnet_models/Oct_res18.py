import numpy as np
import warnings

from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, ZeroPadding2D, AveragePooling2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras.layers import add, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import tensorflow.keras.backend as K
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs

from oct_conv2d import *

import tensorflow as tf

def identity_block(input_tensor, alpha, kernel_size, filters, stage, block, strides = (1,1)):
    
    filters1, filters2 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    
    conv_name_base = 'res' + str(stage) + block + 'conv_branch'
    high_conv_bn_name_base = 'bn' + str(stage) + block + 'high_conv_branch'
    low_conv_bn_name_base = 'bn' + str(stage) + block + 'low_conv_branch'

    high, low = input_tensor

    skip_high, skip_low = input_tensor

    high, low = OctConv2D(filters1, alpha, kernel_size = kernel_size, strides = strides
                    , padding = 'same', name = conv_name_base + '2a')([high, low])
    high = BatchNormalization(axis=bn_axis, name = high_conv_bn_name_base + '2a')(high)
    high = Activation('relu')(high)
    low = BatchNormalization(axis=bn_axis, name = low_conv_bn_name_base + '2a')(low)
    low = Activation('relu')(low)

    high, low = OctConv2D(filters2, alpha, kernel_size = kernel_size, padding = 'same'
                    , name = conv_name_base + '2b')([high, low])
    high = BatchNormalization(axis=bn_axis, name = high_conv_bn_name_base + '2b')(high)
    low = BatchNormalization(axis=bn_axis, name = low_conv_bn_name_base + '2b')(low)

    high = add([high, skip_high])
    low = add([low, skip_low])

    high = Activation('relu')(high)
    low = Activation('relu')(low)

    return [high, low]

def conv_block(input_tensor, alpha, kernel_size, filters, stage, block, strides=(1, 1)):
    
    filters1, filters2 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + 'conv_branch'
    high_conv_bn_name_base = 'bn' + str(stage) + block + 'high_conv_branch'
    low_conv_bn_name_base = 'bn' + str(stage) + block + 'low_conv_branch'

    high, low = input_tensor
    skip_high, skip_low = input_tensor
 
    high, low = OctConv2D(filters1, alpha, kernel_size = kernel_size, padding = 'same'
                , strides= strides, name = conv_name_base + '2a')([high, low])
    high = BatchNormalization(axis=bn_axis, name = high_conv_bn_name_base + '2a')(high)
    high = Activation('relu')(high)
    low = BatchNormalization(axis=bn_axis, name = low_conv_bn_name_base + '2a')(low)
    low = Activation('relu')(low)

    high, low = OctConv2D(filters2, alpha, kernel_size = kernel_size, padding = 'same'
                        , name = conv_name_base + '2b')([high, low])
    high = BatchNormalization(axis=bn_axis, name = high_conv_bn_name_base + '2b')(high)
    low = BatchNormalization(axis=bn_axis, name = low_conv_bn_name_base + '2b')(low)
 
    skip_high = Conv2D(int(filters2 * (1 - alpha)), kernel_size = kernel_size, strides=strides, padding = 'same', name = conv_name_base + '1')(skip_high)
    skip_high = BatchNormalization(axis=bn_axis, name = high_conv_bn_name_base + '1')(skip_high)

    skip_low = Conv2D(int(filters2 * alpha), kernel_size = kernel_size, strides=strides, padding = 'same', name = conv_name_base + '2')(skip_low)
    skip_low = BatchNormalization(axis=bn_axis, name = low_conv_bn_name_base + '2')(skip_low)
 
    # x = add([x, shortcut])
    # x = Activation('relu')(x)
    high = add([high, skip_high])
    low = add([low, skip_low])

    high = Activation('relu')(high)
    low = Activation('relu')(low)

    return [high, low]

def last_OctConv_2_Vanila(input_tensor, filters, alpha):
    
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    high, low = input_tensor

    high_2_high = Conv2D(filters, (3, 3), padding = 'same')(high)
    low_2_high = Conv2D(filters, (3, 3), padding="same")(low)
    low_2_high = Lambda(lambda x: 
                        K.repeat_elements(K.repeat_elements(x, 2, axis=1), 2, axis=2))(low_2_high)
    
    x = add([high_2_high, low_2_high])
    x = BatchNormalization(axis = bn_axis, name = 'bn_last_OctConv_2_Vanila')(x)
    x = Activation('relu')(x)

    return x

def Oct_ResNet18(include_top = False, 
            weights=None,
            alpha = 0,
            input_tensor=None, input_shape=None,
            pooling=None,
            classes=1000):
   
    # if weights not in {'imagenet', None}:
    #     raise ValueError('The `weights` argument should be either '
    #                      '`None` (random initialization) or `imagenet` '
    #                      '(pre-training on ImageNet).')
 
    # if weights == 'imagenet' and include_top and classes != 1000:
    #     raise ValueError('If using `weights` as imagenet with `include_top`'
    #                      ' as true, `classes` should be 1000')
 
 
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=K.image_data_format(),
                                      require_flatten = include_top)
 
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
 
    low = AveragePooling2D(2)(img_input)

    high, low = OctConv2D(64, alpha = alpha, kernel_size = (7, 7), strides = (2, 2))([img_input, low])
    
    high = BatchNormalization(axis=bn_axis, name='high_Oct_bn_conv1')(high)
    high = Activation("relu")(high)
    
    low = BatchNormalization(axis=bn_axis, name='low_Oct_bn_conv1')(low)
    low = Activation("relu")(low)

    high, low = conv_block([high, low], alpha = alpha, kernel_size = (3 , 3), filters = [64, 64], stage = 2, block = 'a')
    high, low = identity_block([high, low], alpha = alpha, kernel_size = (3 , 3), filters = [64, 64], stage = 2, block = 'b')
 
    high, low = conv_block([high, low], alpha = alpha, kernel_size = (3 , 3), filters = [128, 128], stage = 3, block = 'a', strides = (2, 2))
    high, low = identity_block([high, low], alpha = alpha, kernel_size = (3 , 3), filters = [128, 128], stage = 3, block = 'b')

    high, low = conv_block([high, low], alpha = alpha, kernel_size = (3 , 3), filters = [256, 256], stage = 4, block = 'a', strides = (2, 2))
    high, low = identity_block([high, low], alpha = alpha, kernel_size = (3 , 3), filters = [256, 256], stage = 4, block = 'b')

    high, low = conv_block([high, low], alpha = alpha, kernel_size = (3 , 3), filters = [512, 512], stage = 5, block = 'a', strides = (2, 2))
    high, low = identity_block([high, low], alpha = alpha, kernel_size = (3 , 3), filters = [512, 512], stage = 5, block = 'b')
    x = last_OctConv_2_Vanila([high, low], filters = 512, alpha = alpha)
 
    x = AveragePooling2D(name='avg_pool')(x)
 
    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
            x = Dense(classes, activation='softmax', name='resnet18')(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)
 
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
 
    model = Model(inputs, x, name='resnet18')
 
 
    # if weights == 'imagenet':
    #     if include_top:
    #         weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels.h5',
    #                                 WEIGHTS_PATH,
    #                                 cache_subdir='models',
    #                                 md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
    #     else:
    #         weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
    #                                 WEIGHTS_PATH_NO_TOP,
    #                                 cache_subdir='models',
    #                                 md5_hash='a268eb855778b3df3c7506639542a6af')
    #     model.load_weights(weights_path)
    #     if K.backend() == 'theano':
    #         layer_utils.convert_all_kernels_in_model(model)
 
    #     if K.image_data_format() == 'channels_first':
    #         if include_top:
    #             maxpool = model.get_layer(name='avg_pool')
    #             shape = maxpool.output_shape[1:]
    #             dense = model.get_layer(name='fc1000')
    #             layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')
 
    #         if K.backend() == 'tensorflow':
    #             warnings.warn('You are using the TensorFlow backend, yet you '
    #                           'are using the Theano '
    #                           'image data format convention '
    #                           '(`image_data_format="channels_first"`). '
    #                           'For best performance, set '
    #                           '`image_data_format="channels_last"` in '
    #                           'your Keras config '
    #                           'at ~/.keras/keras.json.')
    return model

model = Oct_ResNet18(input_shape = (128, 128, 3), alpha = 0.25, pooling = 'avg', classes = 2)
model.summary()