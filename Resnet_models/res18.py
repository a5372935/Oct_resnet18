import numpy as np
import warnings

from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, ZeroPadding2D, AveragePooling2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import add, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import tensorflow.keras.backend as K
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs

def identity_block(input_tensor, kernel_size, filters, stage, block, strides = (1,1)):
    
    filters1, filters2 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
 
    x = Conv2D(filters1, kernel_size, padding = 'same', strides = strides
                , name=conv_name_base + '2a')(input_tensor)   
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
 
    x = Conv2D(filters2, kernel_size , strides = strides, padding='same'
            , name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
 
    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(1, 1)):
    
    filters1, filters2 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
 
    x = Conv2D(filters1, kernel_size, strides = strides, padding = 'same', name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
 
    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    
    shortcut = Conv2D(filters2, (3, 3), strides=strides, padding = 'same', name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)
 
    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x

def ResNet18(include_top = False, 
            weights=None,
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
 
    #x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    #x = MaxPooling2D((3, 3), strides=(2, 2))(x)
 
    x = conv_block(x, 3, [64, 64], stage=2, block='a')
    x = identity_block(x, 3, [64, 64], stage=2, block='b')
 
    x = conv_block(x, 3, [128, 128], stage=3, block='a', strides=(2, 2))
    x = identity_block(x, 3, [128, 128], stage=3, block='b')
 
    x = conv_block(x, 3, [256, 256], stage=4, block='a', strides=(2, 2))
    x = identity_block(x, 3, [256, 256], stage=4, block='b')
 
    x = conv_block(x, 3, [512, 512], stage=5, block='a', strides=(2, 2))
    x = identity_block(x, 3, [512, 512], stage=5, block='b')
 
    x = AveragePooling2D(name='avg_pool')(x)
 
    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='resnet18')(x)
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

model = ResNet18(input_shape=(64, 64, 3), pooling='avg', classes= 2)
model.summary()