import os
import pandas as pd
from keras import backend as K

# Globals Variable
NOT_PRE_TRAINED_KERAS_MODELS = ["MNIST","CIFAR","CIFAR100","VGG_CIFAR10","VGG_CIFAR100"]

##############################################################################
#  ExtractorConfigBase
#  Configuration object used to feature extraction
# -----------------------------------------------------------------------------

class ExtractorConfig(object):
    '''
    Basic Configuration used for Feature Extraction
    '''
    def __init__(self, root_path='models', model='MODEL',
                 weight_file='weights.h5'):
        '''
        Configuration Constructor
        :param root_path: folder path of  CNN-models
        :param model_name: CNN-model name
        :param weight_file: filename of CNN-model weights
        '''

        # input image dimensions
        self._img_num_channel = None  # placeholder
        self._img_width = None    # placeholder
        self._img_height = None   # placeholder
        self.input_shape = (224, 224, 3)

        self._out_dim = 0          # placeholder

        # data_set config
        self._dataset_path = ''        # dir of data
        self._num_classes = 1000       # placeholder
        self._prototypes_file = None   # path of prototypes dataset
        self._prototypes = None        #
        # extractor config
        self._batch_size = 8  # placeholder

        # FEATURES CONFIG
        # size of single feature vector
        self._features_len = 0  # placeholder
        # features layer
        self._features_index = -1  # placeholder4
        self._features_name = "features"

        # Model



        # Model and weights
        # self.keras_weights = False
        self.model_name = model
        self._weight_file = weight_file
        self._model_file = 'model.json'
        # Extractor models root
        self.models_root = root_path

        # display option
        self.verbose = True
        # output option
        self.output_path = ''
        # update


    #   ------------   Class Properties   --------

    @property
    def model_file(self):
        return self._model_file

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def model_weights(self):
        return self._weight_file

    @property
    def models_root(self):
        return self._models_root

    @property
    def input_shape_2D(self):
        '''
           :return: return the 2D image shape used as model input (compatible with Keras backend setting)
        '''
        return (self._img_width, self._img_height)

    @property
    def input_shape(self):
        '''
            :return: return the 3D image shape used as model input, compatible with Keras backend setting
        '''
        return self._input_shape

    @property
    def img_width(self):
        return self._img_width

    @property
    def img_height(self):
        return self._img_height

    @property
    def features_len(self):
        '''
            :return: return the m-dimensional feature length
        '''
        return self._features_len

    @property
    def features_index(self):
            '''
                :return: return layer index used as image-feature
            '''
            return self._features_index
        # -----------------------------------------------------------------------------------

    @property
    def features_name(self):
        '''
            :return: return layer name used as image-feature
        '''
        return self._features_name

    @property
    def image_channels(self):
        '''
            :return: return the image channel, compatible with Keras backend setting
        '''
        return self._img_num_channel

    @property
    def keras_weights(self):
        '''
            :return: return keras model weights
        '''
        return self._keras_weights

    @property
    def model_name(self):
        '''
         :return: return keras model name
        '''
        return self._model_name

    @property
    def prototypes_file(self):
        '''
         :return: return prototypes dataset path
        '''
        return self._prototypes_file

    #       ------------   Class Setters   --------

    @prototypes_file.setter
    def prototypes_file(self, value):
        '''
        Seeting the filename path of  prototypes dataset
        :param value: weights filename
        :return: None
        '''
        self._prototypes_file = value


    @keras_weights.setter
    def keras_weights(self,value):
        '''
        Seeting the model type and keras_weights filename
        :param value: weights filename
        :return: None
        '''
        self._model_type = 'ImageNet_Keras' if value else 'Custom'
        self._keras_weights = value


    @input_shape.setter
    def input_shape(self, value):
        '''
        Setting the model input_shape
        :param value: shape tuple in format:(width,height,img_num_channel)
        :return: None
        '''

        (self._img_width, self._img_height, self._img_num_channel) = value

        if K.image_data_format() == 'channels_first':
            img_shape = (self._img_num_channel, self._img_width, self._img_height)
        else:
            img_shape = (self._img_width, self._img_height, self._img_num_channel)

        self._input_shape = img_shape

    @models_root.setter
    def models_root(self, root_path='models'):
        '''
                Update the models paths
                :param root_path: folder path of CNN-models
                :return: None
        '''

        if os.path.exists(root_path):
            self._models_root = root_path
            self._model_path = os.path.join(self._models_root, self._model_name)
            self._model_file = os.path.join(self._model_path, 'model', self._model_file)
            self._weight_file = os.path.join(self._model_path, 'model', self._weight_file)
            self._output_path = os.path.join(self._model_path, 'out')
            self._dataset_path = os.path.join(self._model_path, 'data')
            self._prototypes_file = os.path.join(self._dataset_path, 'prototypes.h5')
        else:
            raise Exception("Dir  %s don't exist!!!! " % root_path)

    @model_name.setter
    def model_name(self, name):
            '''
            Setting the model configuration.
            :param name: Keras model name
            :return: None
            '''

            # self._model_type = 'Custom'
            self.keras_weights = False
            self._model_name = name
            self.dataset_path = "DATASETS/" + name
            # ---------------------------------------------------- Custom models
            if name == "MNIST":
                #  Mnist image size
                self.input_shape = (28, 28, 1)
                self._num_classes = 10
                self._features_len = 128
                self._features_name = "features"

            if name == "CIFAR":
                self.input_shape = (32, 32, 3)
                self._num_classes = 10
                self._features_len = 512
                self._features_name = "featuresD"
                self._features_index = -2

            if name == "CIFAR100":
                #  CIFAR image size
                self.input_shape = (32, 32, 3)
                self._num_classes = 100
                self._features_len = 512
                self._features_name = "featuresD"
                self._features_index = -2

            if name == "VGG_CIFAR10":
                #  CIFAR image size
                self.input_shape = (32, 32, 3)
                self._num_classes = 10
                self._features_len = 512
                self._features_name = "featuresD"
                self._features_index = -2

            if name == "VGG_CIFAR100":
                #  CIFAR image size
                self.input_shape = (32, 32, 3)
                self._num_classes = 100
                self._features_len = 512
                self._features_name = "featuresD"
                self._features_index = -2
            # ---------------------------------------------------- ImageNet Keras models
            if name == "VGG16":
                #  DEFINE PARAMETERS
                #  feature_len
                self._features_len = 4096
                self._features_name = "fc2"
                self.keras_weights = True

            if name == "ResNet50":
                #  feature_len
                self._features_len = 2048
                self._features_name = "flatten_1"
                self.keras_weights = True

            if name == "Xception":
                self.input_shape = (299, 299, 3)
                #  feature_len
                self._features_len = 2048
                self._features_name = "avg_pool"
                self.keras_weights = True

            if name == "InceptionV3":
                self.input_shape = (299, 299, 3)
                #  feature_len
                self._features_len = 2048
                self._features_name = "avg_pool"
                self.keras_weights = True

        # -----------------------------------------------------------------------------------
    def print(self):
        '''
        Print the features extractor configuration
        :return: None
        '''

        rows_list = []
        for key, value in vars(self).items():
            row = {}
            # get input row in dictionary format
            row.update({'Attributes': key, 'Values': value})
            rows_list.append(row)
        df = pd.DataFrame(rows_list, columns=('Attributes', 'Values'))
        df = df.sort_values('Attributes')
        print(df.to_string(index=False))

#
# # ***********************************************************************************************
# #  END  Extractor config
# #  ***********************************************************************************************
