from __future__ import print_function
#  keras classification models
from keras.models import Input
from keras.models import model_from_json
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3

import os

# -----------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------Models
AVAILABLE_MODELS = ['VGG16', 'ResNet50', 'Xception', 'InceptionV3']
# assumes K.image_data_format() == 'channels_last'
DEFAULT_SHAPES = {'VGG16':(224, 224, 3), 'ResNet50':(224, 224, 3), 'Xception':(299, 299, 3), 'InceptionV3':(299, 299, 3)}

#  -----------------------------------------------------------------------------------------------

def pre_trained_model(model_name, weights='imagenet', input_shape=None, classes=1000,
                      include_top=True, pooling=None, verbose=True):
    # load model

    if model_name not in AVAILABLE_MODELS:
        raise Exception("Model  %s is not available!!!!. Use : %s " % (model_name, AVAILABLE_MODELS))
    else:
        input_shape = DEFAULT_SHAPES[model_name] if input_shape is None else input_shape
        #  BUILD MODELS
        if model_name == 'VGG16':
                if weights=='imagenet':
                            model =  VGG16(weights='imagenet')
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
        # load weigths
        if weights != 'imagenet':
            if os.path.isfile(weights):
                model.load_weights(weights)
                if verbose:
                    print("{} model weights loaded from {}".format(model_name, weights))
            else:
                raise Exception("Weights file %s don't exist!!!! " % (weights))

        # model.load_weights(weights)
        # if verbose:
        #     print("{} model weights loaded from {}".format(model_name, weights))
        # Freezing layers
        for layer in model.layers:
            layer.trainable = False
            # return model
    return model


#  ------------------------------------------------------------------------------------------------
def model_vgg16(weight_file, input_shape, classes, verbose=True):
    # load model
    model = VGG16(weights=None, input_tensor=None, input_shape=input_shape, pooling=None, classes=classes)
    if verbose:
        print("VGG16 model loaded. \
                    Classes : {}  \
                    Input_shape : {}".format(classes, input_shape))
    # load weigth
    model.load_weights(weight_file)
    if verbose:
        print("VGG16 model weights loaded from {}".format(weight_file))
    # Freezing layers
    for layer in model.layers:
        layer.trainable = False
    # return model
    return model


#  ------------------------------------------------------------------------------------------------
def model_mnist(json_model_file, weight_file, verbose=True):
    #  load json and create model
    model = None
    v_error = False
    if os.path.isfile(json_model_file):
        json_file = open(json_model_file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        if verbose:
            print("MNIST model loaded from {}".format(json_model_file))

    else:
        v_error = True
        raise Exception("File  %s don't exist!!!! " % (json_model_file))

    # Load trained weigth
    if os.path.isfile(weight_file):
        model.load_weights(weight_file)
        if verbose:
            print("MNIST model weights loaded from {}".format(weight_file))
    else:
        v_error = True
        raise Exception("Weights file  %s don't exist!!!! " % (weight_file))

    if v_error:
        return
        # Freezing layers
    for layer in model.layers:
        layer.trainable = False
    return model


# -----------------------------------------------------------------------------------------------------
def local_model(json_model_file, weight_file, verbose=True, freezing=True):
    #  load json and create model
    model = None
    v_error = False
    if os.path.isfile(json_model_file):
        json_file = open(json_model_file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        if verbose:
            print("Model loaded from {}".format(json_model_file))

    else:
        v_error = True
        raise Exception("File  %s don't exist!!!! " % (json_model_file))

    # Load trained weigth
    if os.path.isfile(weight_file):
        model.load_weights(weight_file)
        if verbose:
            print("Model weights loaded from {}".format(weight_file))
    else:
        v_error = True
        raise Exception("Weights file %s don't exist!!!! " % (weight_file))

    if v_error:
        return
    if freezing:
        # Freezing layers
        for layer in model.layers:
            layer.trainable = False
    return model




