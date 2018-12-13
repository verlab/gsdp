from __future__ import print_function
# keras
from keras.applications.imagenet_utils import decode_predictions
# gsdp
from .base import Extractor
from ..utils.imagenet_tools import synset_to_id


class SimpleExtractor(Extractor):
    """
    Class for feature extractor using custom models
    """

    def __init__(self, config):
            """
            Constructor
            """
            # ------------------------------------------------------------
            # Call base init function
            super(SimpleExtractor, self).__init__(config)
    # --------------------------------------------------------------------------------------------

    def top1_highest_prediction(self, img, from_path=False, processed_img=False, verbose=False):
            '''
            Top1 Category prediction
            :param img: PIL image or image path
            :param from_path: flag  to   set  the  image  input   type
            :param processed_img:  preprocessing flag
            :param verbose: logs
            :return: Category prediction
            '''
            if processed_img is False:
                img = self._input_preprocessing(img, from_path=from_path)

            # get the predicted probabilities for each class
            yhat = self.model.predict_classes(img, verbose=verbose)
            return yhat[0]
# ------------------------------------------------------------------------------------------------
#  END  Simple extractor
# ------------------------------------------------------------------------------------------------


class ImageNetExtractor(Extractor):
    """
    Class for feature extractor using Keras ImageNet models
    """

    def __init__(self, config):
            """
            Constructor
            """
            # ------------------------------------------------------------
            # Call base init function
            super(ImageNetExtractor, self).__init__(config)
    # ---------------------------------------------------------------------------------------------

    def top1_highest_prediction(self, img, from_path=False, processed_img=False, verbose=False):
            '''
            Top1 Category prediction
            :param img: PIL image or image path
            :param from_path: flag  to   set  the  image  input   type
            :param processed_img:  pre-processing flag
            :param verbose: logs
            :return: Category prediction
            '''
            if processed_img is False:
                img = self._input_preprocessing(img, from_path=from_path,verbose=verbose)
            # get the predicted probabilities for each class
            yhat = self.model.predict(img, verbose=verbose)
            # retrieve the most likely result, e.g. highest probability
            label = decode_predictions(yhat, top=1)
            label = label[0][0]
            category_index = synset_to_id(label[0])
            if verbose:
                print('%s %s (%.2f%%)' % (str(category_index), label[1], label[2] * 100))
            return category_index
    # ------------------------------------------------------------------------------------------------
#  END  ImageNet extractor
