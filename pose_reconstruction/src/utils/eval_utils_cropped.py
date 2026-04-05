import numpy as np
import json
from os.path import join

def euclid_3d(x, y):
    """
    Calculates the euclidean distance between two column vectors
    :param x: (3, 1) or (3,) vector
    :param y: (3, 1) or (3,) vector
    :return: euclidean distance
    """

    if x.shape.__len__() == 1:
        assert x.shape == (3,) and y.shape == (3,)
    else:
        assert x.shape == (3, 1) and y.shape == (3, 1)

    # Returns euclidean distance
    return np.linalg.norm(x-y)

def calc_IoU(x, y):
    """
    Calculates the intersection over union for two masks
    :param x: Mask1
    :param y: Mask2
    :return: IoU
    """

    assert x.shape.__len__() == 2 and y.shape.__len__() == 2
    assert x.shape == y.shape

    # Converts non-zeros to True
    x = x.astype(bool)
    y = y.astype(bool)

    intersection = x * y
    union = x + y

    return intersection.sum()/float(union.sum())