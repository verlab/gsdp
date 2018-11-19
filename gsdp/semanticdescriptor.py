import os
import h5py
import numpy as np
import gsdp.extractors.extractor as feature_model
import gsdp.extractors.config as config

## Default Values
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) #'/media/DATOS/Dropbox/Deep-Learning/Tutorial_14day/Keras_TensorFlow/GSDP'  #os.path.dirname(os.path.realpath(__file__))

#print(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
MODELS_PATH = os.path.join(ROOT_PATH,'models')           # Keras Models dir
DATASETS_PATH = os.path.join(ROOT_PATH,'data')           # datasets configs
TEST_PATH = os.path.join(ROOT_PATH,'test') # datasets configs

from .tools import compute_plot_signature

class GlobalSemanticDescriptor:
    '''
    GLOBAL SEMANTIC DESCRIPTOR of objects based in category PROTOTYPES (GSDP)
    See paper:
         Vidal Pino, O.; Nascimento, E. R. do; Campos, M. F. M. Prototypicality effects in
         global semantic description of objects. In: Winter Conference on the Applications
         of Computer Vision, 2019, Hawaii. Proceedings...Hawaii: WACV, 2019
    '''

    # ---------------------------------------------------- Create Descriptor Object
    def __init__(self, keras_model_name, models_path=None):
        '''
         Descriptor initialization with the baseline CNN model architecture
              Input:
                    model_name : CNN-model name supported (MNIST,VGG16)
              Output:
                    Create GSDP descriptor object
        '''
        # Model config
        self.prototypes_file = None
        self.prototypes = None
        self.feature_size = 0
        models_path = MODELS_PATH if models_path is None else models_path
        self.model = self._model_init(keras_model_name, models_path)



    # ---------------------------------------------------- Create Model Extractor Object
    def _model_init(self, model_name, models_path, dataset_path=DATASETS_PATH):
        ''' Input :
                   model_name: CNN-model name supported (MNIST,VGG16)
                   models_path: models folder
                   dataset_path: default Images dataset
            Output:
                   Extractor object.
        '''

        model_config = config.ExtractorConfig(root_path=models_path, model_name=model_name)
        # update dataset_np path
        self.prototypes_file = os.path.join(models_path,model_name,'data','prototypes.h5')
        self.prototypes = h5py.File(self.prototypes_file, 'r')
        # build_extractor
        if model_name == 'MNIST':
            model = feature_model.SimpleExtractor(config=model_config)
        else:
            model_config.dataset_path = os.path.join(dataset_path,'ImageNet')
            model = feature_model.ImageNetExtractor(config=model_config)
        return model

    # ---------------------------------------------------- Create Model Extractor Object
    def basefeature(self, img,from_path = False,verbose=True):
        return self.model.feature(img,from_path = from_path,verbose=verbose)[0]

    # ---------------------------------------------------- Create Model Extractor Object
    def describe_category(self,ok):
        return True

    # ---------------------------------------------------- high dimensional semantic description
    def _semantic_representation(self,img,from_path = False,verbose=False, extended_version =True):
        feature, class_idx = self.model.feature_and_prediction(img, from_path=from_path, verbose=verbose)
        difference = None
        if  extended_version:
            abstract_prototype = self.prototypes['mean'][int(class_idx)]
            difference =  np.absolute(feature[0] - abstract_prototype)
        return feature[0], difference, class_idx


    def feature(self, img,from_path = False,verbose=False, extended =True):

        feature, difference, class_idx = self._semantic_representation(img,from_path = from_path,
                                                                       verbose=verbose,
                                                                       extended_version=extended)

        semantic_meaning = compute_plot_signature(self.model, feature, class_idx,plotting=False)

        if extended:  # semantic_meaning + semantic_difference
            semantic_difference = compute_plot_signature(self.model, difference, class_idx,plotting=False,semantic=False)
            return np.concatenate((semantic_meaning, semantic_difference))

        return semantic_meaning

    # def describe(self, image, eps=1e-7):
    #     # compute the Local Binary Pattern representation
    #     # of the image, and then use the LBP representation
    #     # to build the histogram of patterns
    #     lbp = feature.local_binary_pattern(image, self.numPoints,
    #                                        self.radius, method="uniform")
    #     (hist, _) = np.histogram(lbp.ravel(),
    #                              bins=np.arange(0, self.numPoints + 3),
    #                              range=(0, self.numPoints + 2))
    #
    #     # normalize the histogram
    #     hist = hist.astype("float")
    #     hist /= (hist.sum() + eps)
    #
    #     # return the histogram of Local Binary Patterns
    #     return hist

# import tensorflow as tf
# from keras import backend as K
# # session config
# conf = tf.ConfigProto()
# conf.gpu_options.allow_growth = True
# sess = tf.Session(config=conf)
# K.set_session(sess)
# gsdp_desc = GlobalSemanticDescriptor('ResNet50')
# gsdp_desc.model.config._print()