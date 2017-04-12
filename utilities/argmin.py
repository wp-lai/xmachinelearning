def argmin(x):
    """ find the index of the minumum element

    Parameters
    ----------
    x: array-like

    Returns
    -------
    The index of the minumum number

    Examples
    --------
    >>> argmin([10, 0, 20, 30])
    1
    """
    n = len(x)
    return min(range(n), key=lambda i: x[i])
