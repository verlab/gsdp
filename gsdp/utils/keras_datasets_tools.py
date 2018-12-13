from keras import backend as K
from keras.utils import to_categorical

# -------------------------------------------------------------------------------


def keras_input_config(config, x_input, y_input, normalize=False):
    '''
    Setting input format
    :param config: Extractor Configuration
    :param x_input: Inputs
    :param y_input: Labels
    :param normalize: Normalization flag
    :return: Inputs, Labels
    '''

    if x_input is None or len(x_input) == 0:  # if not seq:
        return None, None

    # SETTING INPUT X FORMAT
    if 'channels_first' == K.image_data_format():
        x_input = x_input.reshape(x_input.shape[0],
                                  config.image_channels,
                                  config.img_width,
                                  config.img_height)
    else:
        x_input = x_input.reshape(x_input.shape[0],
                                  config.img_width,
                                  config.img_height,
                                  config.image_channels)

    # CONVERT "INPUT" DATA TO FLOAT AND NORMALIZE
    x_input = x_input.astype('float32')

    if normalize:
        x_input /= 255
    # SETTING INPUT Y FORMAT
    y_input = to_categorical(y_input, config.num_classes)
    return x_input, y_input


# ----------------------------------------------------------------------------------------
def load_data(config, verbose=True):
    '''

    :param config: Extractor Configuration
    :param verbose:
    :return: Keras Data
    '''
    # LOAD DATA from MNIST
    # the data, shuffled and split between train and test sets
    if config.model_name == 'MNIST':
        from keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        normalize = True

    print('-' * 100)
    print('                 LOADING DATA from ', config.model_name)
    print('-' * 100)
    # SETTING INPUT FORMAT
    x_train, y_train = keras_input_config(config,x_train, y_train, normalize=normalize)
    x_test, y_test = keras_input_config(config,x_test, y_test, normalize=normalize)
    input_shape = config.input_shape

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
    return (x_train, x_test), (y_train, y_test)
# ----------------------------------------------------------------------------------------