"""
Mapping a continuous variable to a categorical variable

Examples:
    >>> bins = [20, 40, 60]  # (-INF, 20), [20, 40), [40, 60), [60, INF]
    >>> labels = ["<20", "20~40", "40~60", ">=60"]
    >>> [num_to_range(num, bins, labels) for num in [10, 20, 30, 40, 100]]
    ['<20', '20~40', '20~40', '40~60', '>=60']
"""
from bisect import bisect


def num_to_range(num, bins, labels):
    index = bisect(bins, num)
    return labels[index]
