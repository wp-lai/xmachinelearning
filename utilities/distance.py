from math import sqrt


def dist(p, q):
    """ Euclidean distance for multi-dimensional data

    Examples
    --------
    >>> dist((1, 2), (5, 5))
    5.0
    """
    return sqrt(sum((x - y) ** 2 for x, y in zip(p, q)))
