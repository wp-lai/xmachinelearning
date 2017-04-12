"""
OneHot Encoding

>>> one_hot_encoding([1,0,0,2])
array([[ 0.,  1.,  0.],
       [ 1.,  0.,  0.],
       [ 1.,  0.,  0.],
       [ 0.,  0.,  1.]])
"""
import numpy as np


def one_hot_encoding(array):
    """ One hot encoding for a feature vector

    Parameters
    ----------
    array: 1d array
        A categorical feature vector encoded as [0, 1, 2, ..., n]

    Returns
    -------
    out: numpy.ndarray, shape [n_samples, n_value]
        One hot encoded feature vectors
    """
    array = np.array(array)
    n_value = np.max(array) + 1
    I = np.eye(n_value)
    out = I[array, ]
    return out
