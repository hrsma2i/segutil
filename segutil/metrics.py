import numpy as np


def iou(x: np.ndarray, y: np.ndarray, over_smaller: bool = False) -> float:
    """Intersection over Union

    Parameters
    ----------
    x, y : np.ndarray {0,1 (np.int32)}^(height, width)
        binary masks to compare
    over_smaller : bool, optional
        If this is True, the denominator is the smaller mask's area
        instead of the union, by default False.

    Returns
    -------
    float
    """
    intersection = np.sum(np.logical_and(x, y))
    if over_smaller:
        denominator = min(np.sum(x), np.sum(y))
    else:
        denominator = np.sum(np.logical_or(x, y))
    return intersection / denominator
