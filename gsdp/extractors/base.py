from __future__ import print_function
#  keras
from keras.models import Model
from keras.preprocessing.image import img_to_array
from keras.applications import vgg16, resnet50
from keras.preprocessing.image import load_img

# import keras-models makers
from .models import pre_trained_model, local_model

# import gsdp tools
from ..utils.dump_tools import make_path_exists, loadpklz, savepklz
from .config import NOT_PRE_TRAINED_KERAS_MODELS
from ..utils.metrics import semantic_value, semantic_distance
# import sys utils
import os
import numpy as np
import h5py


class Extractor(object):
    """
     Features Extractor Class
    """
    def __init__(self, config, **kwargs):
            '''
            Basic Features Extractor Constructor
            '''

            # extractor config
            self.config = config
            # Load Main Model
            self._load_main_model(self.config.verbose)
            #  Load feature extractor model
            self._load_model_extractor()
            # weights
            self.W, self.b = self._load_features_weights()
            # prototypes
            self._prototypes = self._load_prototypes()

    @property
    def model_name(self):
            return self.config.model_name

    @property
    def prototypes(self):
        '''
         :return: return prototypes dataset path
        '''
        return self._prototypes

    @property
    def input_shape_2D(self):
        '''
           :return: return the 2D image shape used as model input (compatible with Keras backend setting)
        '''
        return self.config.input_shape_2D

    @property
    def input_shape(self):
        '''
            :return: return the 3D image shape used as model input, compatible with Keras backend setting
        '''
        return self.config.input_shape

    # ---------------------------------------------------------------------------------
    def top1_highest_prediction(self, img, from_path=False, processed_img=False, verbose=False):
            """WRITE ME
            This is a function that define model to feature extract (to
            display additional information)
            """
            print('Warning: Post validation procedure is not set!'
                  ' -- this might be okay')

    # ---------------------------------------------------------------------------------
    def _load_features_weights(self):
            '''
            Load learned feature relevance.
            :return: the feature relevance (weights) learned by softmax layer (last model layer)
            '''
            return self.model.layers[self.config.features_index].get_weights()
    #  -------------------------------------------------------------------------------------------------------

    def _load_prototypes(self):
            '''
            Load learned feature relevance.
            :return: the feature relevance (weights) learned by softmax layer (last model layer)
            '''
            return h5py.File(self.config.prototypes_file, 'r')
    #  -------------------------------------------------------------------------------------------------------

    def _load_main_model(self, verbose=True):
            '''
            Load CNN-classification pre-trained model
            :param verbose: print log
            :return: Keras CNN-classification pre-trained model
            '''

            if verbose:
                print("Loading Model {}...".format(self.model_name))

            try:

                if self.model_name in NOT_PRE_TRAINED_KERAS_MODELS:
                    self.model = local_model(self.config.model_file,
                                             self.config.model_weights,
                                             self.config.verbose)
                else:
                    if self.config.keras_weights:
                        self.model = pre_trained_model(self.config.model_name, verbose=self.config.verbose)
                    else:
                        self.model = pre_trained_model(self.config.model_name,
                                                       self.config.model_weights,
                                                       self.config.input_shape,
                                                       self.config.num_classes,
                                                       self.config.verbose)
            except Exception as inst:
                print('Current configuration with errors.Please check.')
                print('-'*80)
                print(self.config.print())
                print('-'*80)
                print(inst.args)
                raise
    #  -----------------------------------------------------------------------------------------------------

    def _load_model_extractor(self):
            '''
            Base-Model features extractor object. Feature = softmax-feature.
            :return: Base-Model features extractor object
            '''
            f_layer = self.model.get_layer(self.config.features_name)
            self.model_extractor = Model(inputs=self.model.input,
                                         outputs=f_layer.output)
    #  -----------------------------------------------------------------------------------------------------

    def _input_preprocessing(self, img, from_path=False):
            '''
            Pre-processing the input image (img) before feeding into the CNN-model
            :param img: PIL image or image path
            :param from_path: flag to set the image input type
            :return: batch of image(s) to feed the model
            '''
            # load an image from file  gsdp.model.config.input_shape(shape_2D=True)

            grayscale = True if self.config.image_channels == 1 else False
            if from_path:
                    img = load_img(img, grayscale=grayscale, target_size=self.config.input_shape_2D)
            # convert the PIL image to a numpy array
            # IN PIL - image is in (width, height, channel)
            # In Numpy - image is in (height, width, channel)
            # convert the image pixels to a numpy array
            numpy_image = img_to_array(img)

            # Convert the image / images into batch format
            # expand_dims will add an extra dimension to the data at a particular axis
            # We want the input matrix to the network to be of the form (batchsize, height, width, channels)
            # Thus we add the extra dimension to the axis 0.
            image_batch = np.expand_dims(numpy_image, axis=0)

            # prepare the image for the current CNN-model
            if self.config.model_name == 'MNIST':
                return image_batch

            if self.config.model_name == 'VGG16':
                return vgg16.preprocess_input(image_batch)

            if self.config.model_name == 'ResNet50':
                return resnet50.preprocess_input(image_batch)

    # ------------------------------------------------------------------------------------------------
    def feature(self, img, from_path=False, verbose=False):
            '''
            Keras Classification-model feature
            :param img: PIL image or path image
            :param from_path: (True) -> if load img from path dir, else False if img is a PIL image
            :param verbose: print log operations
            :return: CNN-classification feature
            '''
            processed_image = self._input_preprocessing(img, from_path=from_path)
            # get the feature
            return self.model_extractor.predict(processed_image, verbose=verbose)

    # ------------------------------------------------------------------------------------------------
    def feature_and_prediction(self, img, from_path=False, verbose=False):
            '''

            :param img: PIL image or path image
            :param from_path: (True) -> if load img from path dir, else False if img is a PIL image
            :param verbose: print log operations
            :return: Keras CNN- feature, prediction (category index)
            '''

            processed_image = self._input_preprocessing(img, from_path=from_path)
            # get the feature
            feature = self.model_extractor.predict(processed_image.copy(), verbose=verbose)
            # get the predicted probabilities for each class
            category_index = self.top1_highest_prediction(processed_image, processed_img=True, verbose=verbose)
            return feature, category_index

    # ------------------------------------------------------------------------------------------------
    def semantic_value_feature(self, features, category_idx):
            '''
            Return object semantic value from extracted features
            :param features: object feature
            :param category_idx: object category
            :return: object semantic value
            '''
            return semantic_value(self.W[:, category_idx], features[0], self.b[category_idx])

    # ------------------------------------------------------------------------------------------------
    def semantic_value_img(self, img, from_path=False, verbose=False):
            '''
            Return object semantic value from input image
            :param img: PIL image or path image
            :param from_path: (True) -> if load img from path dir, else False if img is a PIL image
            :param verbose: print log operations
            :return: object semantic value
            '''
            features, category_idx = self.feature_and_prediction(img, from_path=from_path, verbose=verbose)
            return self.semantic_value_feature(features, category_idx)
    # ------------------------------------------------------------------------------------------------

    def prototypical_distance_feature(self, features, category_idx):
            '''
            Return object prototypical distance from extracted features
            :param features: object feature
            :param category_idx: object category
            :return: object prototypical distance
            '''

            abstract_prototype = self.prototypes['mean'][int(category_idx)]
            difference = np.absolute(features[0] - abstract_prototype)
            return semantic_distance(self.W[:, category_idx], difference)

    def prototypical_distance_img(self, img, from_path=False, verbose=False):
            '''
            Return object prototypical distance from input image
            :param img: PIL image or path image
            :param from_path: (True) -> if load img from path dir, else False if img is a PIL image
            :param verbose: print log operations
            :return: object semantic value
            '''
            features, category_idx = self.feature_and_prediction(img, from_path=from_path, verbose=verbose)
            return self.prototypical_distance_feature(features, category_idx)

    # ------------------------------------------------------------------------------------------------

    def img_interpretation(self, img, from_path=False, verbose=False):
            '''
            Return semantic value and prototypical distance from input image
            :param img: PIL image or path image
            :param from_path: (True) -> if load img from path dir, else False if img is a PIL image
            :param verbose: print log operations interpretation
            :return: object meaning
            '''
            features, category_idx = self.feature_and_prediction(img, from_path=from_path, verbose=verbose)
            return self.feature_interpretation(features, category_idx)

    def feature_interpretation(self, features, category_idx):
            '''
            Return semantic value and prototypical distance from extracted features
            :param features: object feature
            :param category_idx: object category
            :return: object prototypical distance
            '''
            semantic_value = self.semantic_value_feature(features, category_idx)
            prototypical_distance = self.prototypical_distance_feature(features, category_idx)
            return semantic_value, prototypical_distance

    def dump_config(self, filename='config.zip', dir_path=None):
            '''
            Save the feature extractor configuration
            :param filename: output configuration file
            :param dir_path: output directory
            :return: void
            '''

            if dir_path is None:
                dir_path = self.config.output_path

            make_path_exists(dir_path)
            c_file = os.path.join(dir_path, filename)
            savepklz(self.config, c_file, force_run=True)

    # ------------------------------------------------------------------------------------------------
    def load_config_from_dump(self, filename):
            '''
            Load extractor configuration file
            :param filename: saved extractor configuration
            :return: void
            '''
            if os.path.isfile(filename):
                    self.config = loadpklz(filename, force_run=True)

# *************************************************************************************************
