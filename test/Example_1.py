'''
GSDP Example 1:
GSDP Description of  MNIST Handwritten Digit image using MNIST Keras-model.
'''

from gsdp import GlobalSemanticDescriptor
import os

# 1 - Image Description using image path (using 'from_path' parameter = True)

# setting image path
mnist_img_path = os.path.join('imgs','mnistimg.png')
gsdp_mnist = GlobalSemanticDescriptor('MNIST')

# Uncomment this line to see the descriptor model configuration
#gsdp_mnist.extractor.config._print()

# GSDP feature (for MNIST-Base Model)
gsdp_feature = gsdp_mnist.feature(mnist_img_path,from_path=True)
print(gsdp_feature.shape)


# 2 - Image Description using Pil image (using 'from_path' parameter = False)
from keras.preprocessing.image import load_img
# loading image
mnist_img = load_img(mnist_img_path, grayscale=True,
                     target_size=gsdp_mnist.extractor.input_shape_2D)

# GSDP feature (for MNIST-Base Model)
gsdp_feature2 = gsdp_mnist.feature(mnist_img,from_path=False)
print(gsdp_feature2.shape)
# same feature?
print('Feature difference = ',(gsdp_feature2 - gsdp_feature).sum())



