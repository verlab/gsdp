
from __future__ import print_function
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing.image import img_to_array
from keras.applications import vgg16, resnet50
from .base import Extractor
import numpy as np
import os
from ..utils.imagenet_tools import synsets_map, synset_to_id
from keras.preprocessing.image import load_img


class SimpleExtractor(Extractor):
    """
    Class for feature extractor using custom models
    """

    # ------------------------------------------------------------------------------------------------
    def __init__(self, config):
        """
        Constructor
        """
        # ------------------------------------------------------------
        # Call base init function
        super(SimpleExtractor, self).__init__(config)

    # ----------------------------------------------------------------------------------------------------

    def protoype_real_index(self, data, idx_dataset, class_idx):
        return data[idx_dataset][class_idx]

    # -----------------------------------------------------------------------------------------------------
    def features_tmp_filename(self, path_dir, class_idx):
        return os.path.join(path_dir, str(class_idx))

    # ----------------------------------------------------------------------------------------------------
    def prototype_files(self, dataset, source_path):

        file_index = os.path.join(source_path, 'features', dataset, dataset) + '_features'
        file_stats = os.path.join(source_path, 'f_stat', dataset) + '_stats.h5'
        file_feature = file_index + '.hdf5'

        return file_feature, file_index, file_stats

    # ----------------------------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------------------
    def normalize_CIFAR_to_VGG(self, X_train, X_test):
        # this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the trianing set statistics.
        mean = np.mean(X_train, axis=(0, 1, 2, 3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train - mean) / (std + 1e-7)
        X_test = (X_test - mean) / (std + 1e-7)
        return X_train, X_test

    def normalize_production(self, X_train, X_test):
        # this function is used to normalize instances in production according to saved training set statistics
        # Input: X - a training set
        # Output X - a normalized training set according to normalization constants.

        # these values produced during first training and are general for the standard cifar10 training set normalization
        mean = 121.936
        std = 68.389
        X_train = (X_train - mean) / (std + 1e-7)
        X_test = (X_test - mean) / (std + 1e-7)
        return X_train, X_test

    def load_data(self, verbose=True):
        # LOAD DATA from MNIST
        # the data, shuffled and split between train and test sets
        if self.model_name == 'MNIST':
            from keras.datasets import mnist
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            normalize = True

        if self.model_name == 'CIFAR' or self.model_name == 'VGG_CIFAR10':
            from keras.datasets import cifar10
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            normalize = False

        if self.model_name == 'CIFAR100' or self.model_name == 'VGG_CIFAR100':
            from keras.datasets import cifar100
            (x_train, y_train), (x_test, y_test) = cifar100.load_data()
            normalize = False

        print('-' * 100)
        print('                 LOADING DATA from ', self.model_name)
        print('-' * 100)
        # SETTING INPUT FORMAT
        x_train, y_train = self.keras_input_config(x_train, y_train, normalize=normalize)
        x_test, y_test = self.keras_input_config(x_test, y_test, normalize=normalize)
        input_shape = self.config.input_shape()

        # Normalize with mean for VGG
        if self.model_name == 'VGG_CIFAR10' or self.model_name == 'VGG_CIFAR100':
            print('NORMALIZE DATA')
            print(self.model_name)
            # x_train, x_test = self.normalize_production(x_train, x_test)
            x_train, x_test = self.normalize_CIFAR_to_VGG(x_train, x_test)



        # PRINT DATA FORMAT (FINAL)
        if verbose:
            print(' --------------------  DATA INFO  -------------------------------')
            print(' X train shape      :', x_train.shape)
            print(' Train samples      :', x_train.shape[0])
            print(' X test shape       :', x_test.shape)
            print(' Test samples       :', x_test.shape[0])
            print(' X shape single data:', input_shape)
            print(' Y shape single data:', y_train[0].shape)
            print('-' * 100)
        # init input data
        self.X.fill(x_train, None, x_test)
        self.Y.fill(y_train, None, y_test)
        print(' Done')


# ------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
#  END  extractor MNIST


# ************************************************************************************************
# ------------------------------------------------------------------------------------------------
class ImageNetExtractor(Extractor):
    """
    classdocs
    """

    # ------------------------------------------------------------------------------------------------
    def __init__(self, config):
        """
        Constructor
        """
        # ------------------------------------------------------------
        # Call base init function
        super(ImageNetExtractor, self).__init__(config)
        #self.classes_map = synsets_map()
        # ------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------
    def _preprocess_input(self,img,from_path = False):
        '''
        Perform Pre-processing before feeding the image to the CNN-model
        :param img: PIL image or image path
        :param from_path: flag to set the image input type
        :return: batch of image(s) to feed the model
        '''
        # load an image from file
        if from_path:
               img = load_img(img, target_size=(self.config.img_width, self.config.img_height))
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
        if self.config.model_name=='VGG16':
            return vgg16.preprocess_input(image_batch)

        if self.config.model_name=='ResNet50':
            return resnet50.preprocess_input(image_batch)

    # ------------------------------------------------------------------------------------------------
    def top1_highest_prediction(self,img, from_path = False, processed_img=False,verbose=False):
        if processed_img is False:
              img = self._preprocess_input(img, from_path=from_path)
        # get the predicted probabilities for each class
        yhat = self.model.predict(img)
        # retrieve the most likely result, e.g. highest probability
        label = decode_predictions(yhat, top=1)
        label = label[0][0]
        category_index = synset_to_id(label[0])
        if verbose:
             print('%s %s (%.2f%%)' % (str(category_index),label[1], label[2] * 100))
        return category_index

    # ------------------------------------------------------------------------------------------------
    def feature_and_prediction(self, img, from_path = False, verbose=False):

        processed_image = self._preprocess_input(img,from_path=from_path)
        # get the feature
        feature = self.model_extractor.predict(processed_image.copy(), verbose=verbose)
        # get the predicted probabilities for each class
        category_index = self.top1_highest_prediction(processed_image,processed_img=True,verbose=verbose)
        return feature, category_index

    # ------------------------------------------------------------------------------------------------
    def feature(self, img,from_path = False,verbose=False):
        processed_image = self._preprocess_input(img, from_path=from_path)
        # get the feature
        return self.model_extractor.predict(processed_image, verbose=verbose)

    # ------------------------------------------------------------------------------------------------


    # ------------------------------------------------------------------------------------------------

