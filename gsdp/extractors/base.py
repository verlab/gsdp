from __future__ import print_function
# import Keras

# import dump tools
from ..utils.dump_tools import make_path_exists,loadpklz,savepklz
from .config import MODELS_NOT_USE_IMAGENET
# models makers
from .models import pre_trained_model, local_model
# import sys utils
import os
#  keras functions
from keras import backend as K
from keras.models import Model
from keras.utils  import to_categorical

#-------------------------------------------------------------------------------------
class ExtractorBase(object):
    '''
    Class for Basic Feature Extractor
    '''
    def __init__(self, config, **kwargs):
        '''
        Basic Extractor Constructor
        '''
        # ------------------------------------------------------------
        self.name = 'Extractor_' + config.model_name
        
        # ---------------------------------------------------------------------
        # read config
        self.config = config
        # MODEL
        self.model_name = self.config.model_name
        self.model = None
        self.model_extractor = None
        self.W = None
        self.b = None
        # features

    def _load_main_model(self,verbose = True):
        '''

        :param verbose: print process status
        :return:
        '''
        """WRITE ME
        This is a function that processing and load data (to
        display additional information)
        """
        print('Warning: Post validation procedure is not set!'
              ' -- this might be okay')

    # --------------------------------------------------------------------------------
    def _load_model_extractor(self):
        """WRITE ME
        This is a function that define model to feature extract (to
        display additional information)
        """
        print('Warning: Post validation procedure is not set!'
              ' -- this might be okay')

    # --------------------------------------------------------------------------------
    def feature(self,img):
        """WRITE ME
        This is a function that define model to feature extract (to
        display additional information)
        """
        print('Warning: Post validation procedure is not set!'
              ' -- this might be okay')
    #---------------------------------------------------------------------------------
    def feature_and_prediction(self, img, verbose=False):
        """WRITE ME
        This is a function that define model to feature extract (to
        display additional information)
        """
        print('Warning: Post validation procedure is not set!'
              ' -- this might be okay')
    # --------------------------------------------------------------------------------
    def features(self):
        """WRITE ME
        This is a function that define model to feature extract (to
        display additional information)
        """
        print('Warning: Post validation procedure is not set!'
              ' -- this might be okay')

    # --------------------------------------------------------------------------------
    def feature_from_path(self):
        """WRITE ME
        This is a function that define model to feature extract (to
        display additional information)
        """
        print('Warning: Post validation procedure is not set!'
              ' -- this might be okay')

    # --------------------------------------------------------------------------------
    def feature_from_paths(self):
            """WRITE ME
            This is a function that define model to feature extract (to
            display additional information)
            """
            print('Warning: Post validation procedure is not set!'
                  ' -- this might be okay')
    # ---------------------------------------------------------------------------------
    def feature_save(self):
        """WRITE ME
        This is a function that run feature extraction a (to
        display additional information)
        """
        print('Warning: Post validation procedure is not set!'
              ' -- this might be okay')

    # ---------------------------------------------------------------------------------
    def feature_extractor(self):
        """WRITE ME
        This is a function that run feature extraction a (to
        display additional information)
        """
        print('Warning: Post validation procedure is not set!'
              ' -- this might be okay')
    # ---------------------------------------------------------------------------------
    def dump_config(self,filename='config.zip',dir_path = None):

        if dir_path is None:
            dir_path = self.config.output_path

        make_path_exists(dir_path)
        c_file = os.path.join(dir_path,filename)
        savepklz(self.config, c_file, force_run=True)

    # ---------------------------------------------------------------------------------
    def load_config_from_dump(self, filename):
        if os.path.isfile(filename):
              self.config = loadpklz(filename,force_run=True)


###########################################################################################
# END  ExtractorBase
###########################################################################################

class Extractor(ExtractorBase):
    """
     Class to Feature Extraction
    """

    #  ------------------------------------------------------------------------------------------------
    def __init__(self, config):
        """
            Constructor
        """
        #  ------------------------------------------------------------
        #  Call base init function
        super(Extractor, self).__init__(config)
        #  ------------------------------------------------------------
        #  Load Main Model
        self._load_main_model()
        #  ------------------------------------------------------------
        #  Load feature extractor model
        self._load_model_extractor()
        #  init W and b
        self.W, self.b = self._load_features_weights()
        self.class_map = None

    #  -------------------------------------------------------------------------------------------------------
    def _load_features_weights(self):
        '''

        :return: the feature relevance used in softmax layer
        '''
        return self.model.layers[self.config.features_idx_w].get_weights()

    #  -------------------------------------------------------------------------------------------------------
    def _load_main_model(self, keras_weights=False,verbose=True):
        '''
        LOAD PRE-TRAINED CNN-MODEL
        :param keras_weights: If True load keras imagenet weights, else load imagenet weights saved in model path
        :param verbose: print log
        :return: CNN-model
        '''

        #  LOAD PRE-TRAINED MODEL

        if verbose:
            print("Loading Model {}...".format(self.model_name))

        try:

            if self.model_name in MODELS_NOT_USE_IMAGENET:
                self.model = local_model(self.config.model_file,
                                         self.config.weight_file,
                                         verbose)
            else:
                if keras_weights:
                    self.model = pre_trained_model(self.config.model_name, verbose=verbose)
                else:
                    self.model = pre_trained_model(self.config.model_name,
                                                   self.config.weight_file,
                                                   self.config.input_shape(),
                                                   self.config.num_classes,
                                                   verbose)
        except Exception as inst:
            print('Current configuration with errors.Please check.')
            print('-'*80)
            print(self.config._print())
            print('-'*80)
            print(inst.args)
            raise

    #  -----------------------------------------------------------------------------------------------------
    def _load_model_extractor(self):
        f_layer = self.model.get_layer(self.config.features_name)
        self.model_extractor = Model(inputs=self.model.input,
                                     outputs=f_layer.output)
    #  -----------------------------------------------------------------------------------------------------
    def load_data(self, verbose=True):
        #  LOAD DATA from dataset
        #  the data, shuffled and split between train and test sets
        """WRITE ME
        This is a function that run feature extraction a (to
        display additional information)
        """
        print('Warning: Post validation procedure is not set!'
              ' -- this might be okay')

    #  --------------------------------------------------------------------------------------------------
    #  *****************   FEATURES FUNCT. (Extraction, save, and load) *********************************
    #  SETTING INPUT FORMAT
    def keras_input_config(self, x_input, y_input, normalize=False):

        if x_input is None or len(x_input) == 0:  # if not seq:
            return None, None

        # SETTING INPUT X FORMAT
        if 'channels_first' == K.image_data_format():
            x_input = x_input.reshape(x_input.shape[0],
                                      self.config.num_channel,
                                      self.config.img_width,
                                      self.config.img_height)
        else:
            x_input = x_input.reshape(x_input.shape[0],
                                      self.config.img_width,
                                      self.config.img_height,
                                      self.config.num_channel)

        # CONVERT "INPUT" DATA TO FLOAT AND NORMALIZE
        x_input = x_input.astype('float32')

        if normalize:
            x_input /= 255
        # SETTING INPUT Y FORMAT
        y_input = to_categorical(y_input, self.config.num_classes)
        return x_input, y_input

    #----------------------------------------------------------------------------------------------------
    def load_file_index(self,file):
      file = file + '.zip'
      return loadpklz(file ,force_run=True)

    #----------------------------------------------------------------------------------------------------

#
# *************************************************************************************************
