# gsdp imports
import gsdp.extractors.extractor as feature_model
import gsdp.extractors.config as config
from .tools import compute_plot_signature
# sys
import os
import numpy as np


# Default Values Paths
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
MODELS_PATH = os.path.join(ROOT_PATH, 'models')        # Keras Models dir
DATASETS_PATH = os.path.join(ROOT_PATH, 'data')        # Datasets configs
TEST_PATH = os.path.join(ROOT_PATH, 'test')            # Examples

# feature config
FEATURE_LENGTH = {'MNIST': [32, 128], 'VGG16': [256, 1024], 'ResNet50': [128, 512]}
SQUARE_AUX_MATRIX = {'MNIST': [8, 4], 'VGG16': [16, 8], 'ResNet50': [16, 8]}


class GlobalSemanticDescriptor:
    '''
    GLOBAL SEMANTIC DESCRIPTOR of objects based in category PROTOTYPES (GSDP)
    See paper:
         Vidal Pino, O.; Nascimento, E. R. do; Campos, M. F. M. Prototypicality effects in
         global semantic description of objects. In: Winter Conference on the Applications
         of Computer Vision, 2019, Hawaii. Proceedings...Hawaii: WACV, 2019
    '''

    # ------------------------------------------ Create Descriptor Object
    def __init__(self, keras_model_name, models_path=None, size_option=1):
            '''
            Descriptor initialization with the baseline CNN model architecture
                  Input:
                        model_name: CNN-model name supported (MNIST, VGG16)
                  Output:
                        Create GSDP descriptor object

            :param keras_model_name: Keras model name
            :param models_path: Models root path
            '''
            models_path = MODELS_PATH if models_path is None else models_path
            self.extractor = self._model_init(keras_model_name, models_path, size_option)
            self.prototypes = self.extractor.prototypes

    # ------------------- Create Model Extractor Object
    def _model_init(self, model_name, models_path, feature_opt=1):
            ''' Input :
                       model_name: CNN-model name supported (MNIST,VGG16)
                       models_path: models folder
                       dataset_path: default Images dataset
                Output:
                       Extractor object.
            '''

            model_config = config.ExtractorConfig(root_path=models_path, model=model_name)
            self._base_model = model_name

            # build_extractor
            if model_name == 'MNIST':
                model = feature_model.SimpleExtractor(config=model_config)
            else:
                model_config.dataset_path = os.path.join(DATASETS_PATH, 'ImageNet')
                model = feature_model.ImageNetExtractor(config=model_config)

            # feature_config
            self.feature_size = feature_opt
            return model

    @property
    def feature_size(self):
            return self._feature_size

    @property
    def base_model(self):
            return self._base_model

    @property
    def aux_matrix_dim(self):
            return self._aux_matrix_dim

    @feature_size.setter
    def feature_size(self, option):
            self._feature_size = FEATURE_LENGTH[self.base_model][option - 1]
            self._aux_matrix_dim = SQUARE_AUX_MATRIX[self.base_model][option - 1]

    # ----------------------------------------------- high dimensional semantic description

    def _semantic_representation(self, img, from_path=False, verbose=False, extended_version=True):
            '''
            Build Semantic Representation
            :param img: PIL image or path image
            :param from_path: Image flag. (True) -> if load img from path dir, else False if img is a PIL image
            :param verbose:  log
            :param extended_version:
            :return: img feature, difference, category index
            '''
            feature, class_idx = self.extractor.feature_and_prediction(img, from_path=from_path, verbose=verbose)
            difference = None
            if extended_version:
                abstract_prototype = self.prototypes['mean'][int(class_idx)]
                difference = np.absolute(feature[0] - abstract_prototype)
            return feature[0], difference, class_idx
    # ------------------------------------------------ Base Model Features Extraction

    def base_feature(self, img, from_path=False, verbose=True):
            '''
            Base Model feature
            :param img: PIL image or path image
            :param from_path: Image flag.
            :param verbose: log
            :return: Base Model feature
            '''
            return self.extractor.feature(img, from_path=from_path, verbose=verbose)[0]
    # ----------------------------------------------------   single GSDP feature extraction

    def feature(self, img, from_path=False, verbose=False, extended=True):
            '''
            Global Semantic feature
            :param img: PIL image or path image
            :param from_path: Image format flag.
            :param verbose:
            :param extended:
            :return: GSDP feature
            '''
            feature, difference, category_idx = self._semantic_representation(img, from_path=from_path,
                                                                              verbose=verbose, extended_version=extended)

            semantic_meaning = compute_plot_signature(self.extractor, feature, category_idx, aux_matrix_dim=self.aux_matrix_dim, plotting=False)
            if extended:  # semantic_meaning + semantic_difference
                semantic_difference = compute_plot_signature(self.extractor, difference, category_idx,
                                                             aux_matrix_dim=self.aux_matrix_dim, plotting=False, semantic=False)
                return np.concatenate((semantic_meaning, semantic_difference))

            return semantic_meaning

    # ---------------------------------------------------- Create Model Extractor Object
#    def describe_category(self, ok):
#            return True
