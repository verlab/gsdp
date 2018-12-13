from __future__ import print_function
#  keras classification models
from keras.models import Input
from keras.models import model_from_json
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3

import os

# --------------------------------------------------------------------------------------------Models
KERAS_MODELS = ['VGG16', 'ResNet50', 'Xception', 'InceptionV3']
# assumes K.image_data_format() == 'channels_last'
DEFAULT_SHAPES = {'VGG16': (224, 224, 3), 'ResNet50': (224, 224, 3), 'Xception': (299, 299, 3), 'InceptionV3': (299, 299, 3)}
#  -----------------------------------------------------------------------------------------------


def pre_trained_model(model_name, weights='imagenet', input_shape=None, classes=1000,
                      include_top=True, pooling=None, verbose=True):
    '''
    Create Keras pre-trained model.
    :param model_name: Keras pre-trained model name
    :param weights: model weights
    :param input_shape: model input
    :param classes: categories number
    :param include_top:
    :param pooling:
    :param verbose:
    :return: model
    '''

    # load model
    if model_name not in KERAS_MODELS:
        raise Exception("Model  %s is not available!!!!. Use : %s " % (model_name, KERAS_MODELS))
    else:
        input_shape = DEFAULT_SHAPES[model_name] if input_shape is None else input_shape
        #  BUILD MODELS
        if model_name == 'VGG16':
                if weights == 'imagenet':
                            model = VGG16(weights='imagenet')
                else:
                            model = VGG16(include_top=include_top,
                                          weights=None,
                                          input_tensor=None,
                                          input_shape=input_shape,  # default: (224, 224, 3)
                                          pooling=pooling,
                                          classes=classes)
        if model_name == 'ResNet50':
                if weights == 'imagenet':
                            model = ResNet50(weights='imagenet')
                else:
                            model = ResNet50(include_top=include_top,
                                             weights=None,
                                             input_tensor=None,
                                             input_shape=input_shape,  # default: (224, 224, 3)
                                             pooling=pooling,
                                             classes=classes)
        if model_name == 'Xception':
            model = Xception(include_top=include_top,
                             weights=None,
                             input_tensor=None,
                             input_shape=input_shape,  # default: (299, 299, 3)
                             pooling=pooling,
                             classes=classes)

        if model_name == 'InceptionV3':
            input_tensor = Input(shape=input_shape)  # this assumes K.image_data_format() == 'channels_last'
            model = InceptionV3(include_top=include_top,
                                weights=None,
                                input_tensor=input_tensor,
                                input_shape=input_shape,  # default: (299, 299, 3)
                                pooling=pooling,
                                classes=classes)
        # model loaded
        if verbose:
            print("{} model loaded.     Classes : {}  \
                            Input_shape : {}".format(model_name, classes, input_shape))
        # load weights
        if weights != 'imagenet':
            if os.path.isfile(weights):
                model.load_weights(weights)
                if verbose:
                    print("{} model weights loaded from {}".format(model_name, weights))
            else:
                raise Exception("Weights file %s don't exist!!!! " % weights)

        # Freezing layers
        for layer in model.layers:
            layer.trainable = False
            # return model
    return model


# -----------------------------------------------------------------------------------------------------
def local_model(json_model_file, weight_file, verbose=True, freezing=True):
    '''
    Create Custom pre-trained model.
    :param json_model_file: model file (.json)
    :param weight_file: model weights
    :param verbose: log
    :param freezing: Freezing layers flag
    :return: model
    '''

    #  load json and create model
    if os.path.isfile(json_model_file):
        json_file = open(json_model_file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        if verbose:
            print("Model loaded from {}".format(json_model_file))

    else:
        raise Exception("File  %s don't exist!!!! " % json_model_file)

    # Load trained weights
    if os.path.isfile(weight_file):
        model.load_weights(weight_file)
        if verbose:
            print("Model weights loaded from {}".format(weight_file))
    else:
        raise Exception("Weights file %s don't exist!!!! " % weight_file)

    if freezing:
        # Freezing layers
        for layer in model.layers:
            layer.trainable = False
    return model
# -----------------------------------------------------------------------------
