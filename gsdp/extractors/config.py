import os
import pandas as pd
from keras import backend as K

# Globals Variable
MODELS_NOT_USE_IMAGENET = ["MNIST","CIFAR","CIFAR100","VGG_CIFAR10","VGG_CIFAR100"]

##############################################################################
#  ExtractorConfigBase
#  Configuration object used to feature extraction
# -----------------------------------------------------------------------------


class ExtractorConfigBase(object):
    '''
    Basic Configuration used for Feature Extraction
    '''
    def __init__(self, root_path='models', model_name='MODEL', weight_file='weights.h5'):
        '''
        Config Constructor
        :param root_path: folder path of  CNN-models
        :param model_name: CNN-model name
        :param weight_file: filename of CNN-model weights
        '''

        # input image dimensions
        self.num_channel = 0  # placeholder
        self.img_width = 0  # placeholder
        self.img_height = 0  # placeholder
        # patch
        # self.patch_height = 0  # placeholder
        # self.patch_width = 0  # placeholder

        self.out_dim = 0  # placeholder
        # data_set config
        self.dataset_path = ''  # dir of data
        self.num_classes = 0  # placeholder
        self.idx_data = {'train': 0, 'val': 1, 'test': 2}
        # extract config

        self.batch_size = 0  # placeholder
        # self.learning_rate = 0.01
        # self.momentum = 0.9
        #self.num_epochs = 30  # placeholder
        # save results regardless of the validation result for this amount of
        # epochs
        #self.min_epoch = 5
        # Model
        self.models_root = root_path
        self.model_type = 'CNN'
        self.model_name = model_name
        self.model_path = '{}/{}'.format(self.models_root, self.model_name)
        self.model_file = 'model.json'
        self.weight_file = weight_file

        # features_config
        # size of single feature vector
        self.features_len = 0  # placeholder
        # name of feature layers
        self.features_index = 0  # placeholder4

        # display option
        self.process_verbose = True
        # output option
        self.output_path = ''
        # update
        self._modelpath_init(model_name, root_path)

    # -----------------------------------------------------------------------------------
    def _modelpath_init(self, model_name="MODEL", root_path='models'):
        '''
        Init models paths
        :param model_name: CNN-model name
        :param root_path: folder path of CNN-models
        :return: None
        '''

        # init path values
        self.model_name = model_name
        if not os.path.exists(root_path):
            raise Exception("Dir  %s don't exist!!!! " % root_path)
        else:
            self._modelpath_upd(root_path)

    # -----------------------------------------------------------------------------------
    def _modelpath_upd(self, root_path='models'):
        '''
        Update the models paths
        :param root_path: folder path of CNN-models
        :return: None
        '''
        if os.path.exists(root_path):
            self.models_root = root_path
            self.model_path = os.path.join(self.models_root, self.model_name)
            self.model_file = os.path.join(self.model_path, 'model', self.model_file)
            self.weight_file = os.path.join(self.model_path, 'model', self.weight_file)
            self.output_path = os.path.join(self.model_path, 'out')
            self.dataset_path = os.path.join(self.model_path, 'data')
        else:
            raise Exception("Dir  %s don't exist!!!! " % root_path)

    # -----------------------------------------------------------------------------------
    def _print(self):
        '''
        Print the feature extractor configuration
        :return: None
        '''

        rows_list = []
        for key, value in vars(self).items():
            row = {}
            # get input row in dictionary format
            row.update({'attributes': key, 'values': value})
            rows_list.append(row)
        df = pd.DataFrame(rows_list, columns=('attributes', 'values'))
        df = df.sort_values('attributes')
        print(df.to_string(index=False))

    # -----------------------------------------------------------------------------------
    def input_shape(self):
        '''
        :return: return the image shape used as model input, compatible with Keras backend used
        '''
        if K.image_data_format() == 'channels_first':
            input_shape = (self.num_channel, self.img_width, self.img_height)
        else:
            input_shape = (self.img_width, self.img_height, self.num_channel)
        return input_shape
        # -----------------------------------------------------------------------------------

    def update_input_shape(self, img_width, img_height, num_channel):
        '''
        Update the image shape used as model input
        :param img_width: Image width
        :param img_height: Image height
        :param num_channel: Image channel
        :return: None
        '''
        self.img_width = img_width
        self.img_height = img_height
        self.num_channel = num_channel


###########################################################################################
# END  extractor config
###########################################################################################


class ExtractorConfig(ExtractorConfigBase):
    def __init__(self, root_path='models', model_name='MODEL', weight_file='weights.h5'):
        '''
        Config Constructor
        :param root_path: folder path of  CNN-models
        :param model_name: CNN-model name
        :param weight_file: filename of CNN-model weights
        '''
        #  Call base init function
        super(ExtractorConfig, self).__init__(root_path, model_name, weight_file)

        #  DEFINE PARAMETERS
        # Update name and path
        #  patch  28, 28
        #self.patch_height = 28
        #self.patch_width = 28
        #  run parameters
        self.batch_size = 32
        #self.num_epochs = 12
        #  default values
        self.model_name = model_name
        self.update_input_shape(224, 224, 3)
        self.num_classes = 1000
        self.features_idx_w = -1
        self.model_type = 'Custom'
        # ---------------------------------------------------- Custom models
        if model_name == "MNIST":
            #  Mnist image size
            self.update_input_shape(28, 28, 1)
            #  dataset dim parameters
            self.num_classes = 10
            #  feature_len
            self.features_len = 128
            self.dataset_path = "DATASETS/MNIST"
            self.features_name = "features"

        if model_name == "CIFAR":
            #  CIFAR image size
            self.update_input_shape(32, 32, 3)
            #  dataset dim parameters
            self.num_classes = 10
            #  feature_len
            self.features_len = 512
            self.dataset_path = "DATASETS/CIFAR"
            self.features_name = "featuresD"
            self.features_idx_w = -2

        if model_name == "CIFAR100":
            #  CIFAR image size
            self.update_input_shape(32, 32, 3)
            #  dataset dim parameters
            self.num_classes = 100
            #  feature_len
            self.features_len = 512
            self.dataset_path = "DATASETS/CIFAR"
            self.features_name = "featuresD"
            self.features_idx_w = -2

        if model_name == "VGG_CIFAR10":
            #  CIFAR image size
            self.update_input_shape(32, 32, 3)
            #  dataset dim parameters
            self.num_classes = 10
            #  feature_len
            self.features_len = 512
            self.dataset_path = "DATASETS/CIFAR"
            self.features_name = "featuresD"
            self.features_idx_w = -2

        if model_name == "VGG_CIFAR100":
            #  CIFAR image size
            self.update_input_shape(32, 32, 3)
            #  dataset dim parameters
            self.num_classes = 100
            #  feature_len
            self.features_len = 512
            self.dataset_path = "DATASETS/CIFAR"
            self.features_name = "featuresD"
            self.features_idx_w = -2
        # ---------------------------------------------------- ImageNet Keras models
        if model_name == "VGG16":
            #  DEFINE PARAMETERS
            #  feature_len
            self.features_len = 4096
            self.features_name = "fc2"
            self.model_type = 'ImageNet Keras'

        if model_name == "ResNet50":
            #  feature_len
            self.features_len = 2048
            self.features_name = "flatten_1"
            self.model_type = 'ImageNet Keras'

        if model_name == "Xception":
            self.update_input_shape(299, 299, 3)
            #  feature_len
            self.features_len = 2048
            self.features_name = "avg_pool"
            self.model_type = 'ImageNet Keras'

        if model_name == "InceptionV3":
            self.update_input_shape(299, 299, 3)
            #  feature_len
            self.features_len = 2048
            self.features_name = "avg_pool"
            self.model_type = 'ImageNet Keras'

            #  ------------------------------------------------------------------------------------

# ***********************************************************************************************
#  END  Extractor config
#  ***********************************************************************************************
