'''
GSDP Example 3:
GSDP Image Description using VGG16 Keras-model.
'''
import tensorflow as tf
from keras import backend as K

from gsdp import GlobalSemanticDescriptor
from gsdp import TEST_PATH
#from test import img as img_test
from keras.preprocessing.image import load_img

# session config
conf = tf.ConfigProto()
conf.gpu_options.allow_growth = True
sess = tf.Session(config=conf)
K.set_session(sess)

# Model
gsdp = GlobalSemanticDescriptor('VGG16')
# print extractor config
# gsdp.extractor.config.print()

# Load PIL image
img_test = TEST_PATH + '/imgs/image.jpg'
image = load_img(img_test, target_size=gsdp.extractor.input_shape_2D)

# Base Model feature extraction
VGG16_feature = gsdp.base_feature(image,verbose=False)
print('VGG16 feature shape: ', VGG16_feature.shape)

# GSDP feature extraction
GSDP_feature = gsdp.feature(image)
print('GSDP feature shape: ', GSDP_feature.shape)