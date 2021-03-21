from typing import Sequence
import numpy as np
from PIL import Image

from segmentation.visualizations import voc_colormap


def masks_to_segmap(masks: np.ndarray, class_ids: np.ndarray) -> np.ndarray:
    """Compress binary masks to a segmentatoin map for a particular image

    Args:
        masks     (np.ndarray; {0, 1}^(height, width, #rles_i)): binary masks
        class_ids (np.ndarray;      C^(#rles_i, )):
            C: {0, 1, ..., #classes-1}
                0: must be background
            #rles_i: the number of RLE annotations for an image `i`

    Returns:
        segmap (np.ndarray; C^(height, width)): a segmentation map
    """
    segmap = np.max(class_ids * masks, axis=-1).astype(np.uint8)
    # segmap: (height, width)

    return segmap


def segmap_to_pil(
    segmap: np.ndarray,
    all_class_ids: Sequence[int],
) -> Image.Image:
    """[summary]

    Args:
        segmap    (np.ndarray; C^(height, width)): a segmentation map
            C: {0, 1, ..., #classes-1}
                0: must be background
        all_class_ids (np.ndarray;  C^(#C, )):

    Returns:
        Image.Image: [description]
    """
    palette = voc_colormap(all_class_ids)

    img = Image.fromarray(segmap).convert("P")
    img.putpalette(palette.astype(np.uint8))
    return img
