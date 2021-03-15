from typing import Sequence, List
import numpy as np
from PIL import Image

from segmentation.visualizations import voc_colormap


def masks_to_segmap(masks: np.ndarray, class_ids: np.ndarray) -> np.ndarray:
    """Compress binary masks to a segmentatoin map for a particular image

    Args:
        masks        (np.ndarray; {0, 1}^(height, width, #rles_i)): binary masks
        class_ids (np.ndarray;      C^(#rles_i, )):
            C: {0, 1, ..., #classes-1}
            #rles_i: the number of RLE annotations for an image `i`

    Returns:
        segmap (np.ndarray; C'^(height, width)): a segmentation map
            C': {0, 1, ..., #classes-1, 255}
                255: background
    """
    # `+1` avoids from collision beween class 0 and background
    segmap = np.max((class_ids + 1) * masks, axis=-1).astype(np.uint8)
    # segmap: (height, width)

    # `-1` shifts `0` to `255` because segmap is np.uint8
    return segmap - 1


def segmap_to_pil(
    segmap: np.ndarray,
    class_ids: Sequence[int],
) -> Image.Image:
    """[summary]

    Args:
        segmap    (np.ndarray; C'^(height, width)): a segmentation map
            C': {0, 1, ..., #classes-1, 255}
                255: background
        class_ids (np.ndarray;  C^(#rles_i, )):
            C:  {0, 1, ..., #classes-1}
            #rles_i: the number of RLE annotations for an image `i`

    Returns:
        Image.Image: [description]
    """
    background = np.array([255, 255, 255]).reshape(1, -1)
    palette = np.concatenate(
        (voc_colormap(class_ids), background),
        axis=0,
    )

    img = Image.fromarray(segmap).convert("P")
    img.putpalette(palette.astype(np.uint8))
    return img
