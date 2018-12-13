'''
GSDP Example 2:
GSDP Description of  MNIST Handwritten Digit image using MNIST Keras-model.
Use Keras MNIST Dataset
'''

# session config
import tensorflow as tf
from keras import backend as K
K.set_session(tf.Session())

# import gsdp utils
from gsdp.utils.keras_datasets_tools import load_data
from gsdp import GlobalSemanticDescriptor
gsdp_d = GlobalSemanticDescriptor('MNIST')
import numpy as np
from time import time
from tqdm import tqdm
#gsdp_d.extractor.config._print()
import os

(x_train, x_test), (y_train, y_test) = load_data(gsdp_d.extractor.config)
gsdp_features = []
t_start = time()

for img in tqdm(x_train[:1000]):
     gsdp_feature = gsdp_d.feature(img)
     gsdp_features.append(gsdp_feature)
gsdp_features = np.row_stack(gsdp_features)
print(gsdp_features.shape)
print('Time:', time()-t_start)

