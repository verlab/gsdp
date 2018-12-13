# ----------- distance metrics between a and b
import numpy as np
# -----------------------------------------------------
# L2 Family
# -----------------------------------------------------


def simpleL2(a, b):
    '''
    :param a: feature
    :param b: feature
    :return: L2 metric
    '''
    q = a-b
    return np.sqrt((q*q).sum())
# -----------------------------------------------------


def weightedL2(a, b, w):
    '''
    Weighted Euclidean Distance
    :param a: feature
    :param b: feature
    :param w: feature relevance
    :return: Weighted Euclidean Distance
    '''
    q = a-b
    return np.sqrt((w*q*q).sum())
# -----------------------------------------------------


def normalizedL2(a, b, s):
    '''
    Normalized Euclidean distance
    :param a: feature
    :param b: feature
    :param s: features std
    :return: distance
    '''
    q = a-b
    return np.sqrt((q*q/s*s).sum())

# -----------------------------------------------------
# L1 family
# -----------------------------------------------------


def simpleL1(a, b):
    '''
    :param a: feature
    :param b: feature
    :return: L1 Norm
    '''
    q = a-b
    return np.absolute(q).sum()
# -----------------------------------------------------


def absoluteWxL1(a, b, w):
    '''
    :param a: feature
    :param b: feature
    :param w: feature relevance
    :return:  Absolute Weighted L1
    '''
    q = a-b
    return np.absolute(w*q).sum()
# -----------------------------------------------------


def absoluteWxL1_N(a, b, w):
    '''

    :param a: feature
    :param b: feature
    :param w: feature relevance
    :return : absolute normalized Weighted L1
    '''
    q = a-b
    sw = np.absolute(w).sum()
    return np.absolute(w*q).sum()/sw


# -----------------------------------------------------
def semantic_value(w, f, b):
    '''
    Semantic Value
    :param w: learned weight
    :param f: feature
    :param b: bias
    :return: Semantic value
    '''
    return (w*f).sum() + b
# -----------------------------------------------------


def semantic_distance(w, f):
    '''
    Semantic distance
    :param w: learned weight
    :param f: feature
    :return: Semantic distance
    '''
    return (np.absolute(w)*f).sum()

# -----------------------------------------------------
