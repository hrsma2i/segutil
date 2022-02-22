from typing import Sequence, List, Dict, Any, Union

import numpy as np
from PIL import Image
from pycocotools import mask as mutils

from segutil.visualizations import voc_colormap
from segutil.types import Polygon, COCORLE


def decode_mask(
    encoded_mask: Union[COCORLE, List[Polygon]],
    height: int = None,
    width: int = None,
) -> np.ndarray:
    """

    Parameters
    ----------
    encoded_mask : Union[COCORLE, List[Polygon]]
    height : int, optional
    width : int, optional
        The height and width of the image.
        This is necessary for Polygon, by default None

    Returns
    -------
    np.ndarray {0,1 (np.int32)}^(height, width)
        a binary mask
    """
    if isinstance(encoded_mask, list):
        return decode_polygon(encoded_mask, height, width)
    elif isinstance(encoded_mask, dict) and "counts" in encoded_mask.keys():
        return decode_rle(encoded_mask)
    else:
        raise ValueError(f"invalid encoding type: {type(encoded_mask)}")


def decode_polygon(polygons: List[Polygon], height: int, width: int) -> np.ndarray:
    return np.max(mutils.decode(mutils.frPyObjects(polygons, height, width)), axis=2)


def decode_rle(rle: COCORLE) -> np.ndarray:
    return mutils.decode(rle)


def masks_to_segmap(masks: List[np.ndarray], category_ids: List[int]) -> np.ndarray:
    """Compress binary masks to a segmentatoin map for a particular image

    Args:
        masks     (List[np.ndarray]; [{0, 1}^(height, width) * #RLEs]): binary masks
            #RLEs: the number of RLEs for a single image.
        category_ids (List[int]): A list of category ids for each binary mask.
            This ranges in {0, 1, ..., #categories}.
            0: must be background
            #categories: The number of all categories.

    Returns:
        segmap (np.ndarray; C^(height, width)): a segmentation map
    """
    masks = np.array(masks)
    # (#masks, height, width)
    category_ids = np.array(category_ids).reshape(-1, 1, 1)
    # (#masks, 1, 1)
    segmap = np.max(category_ids * masks, axis=0).astype(np.int32)
    # segmap: (height, width)

    return segmap


def segmap_to_pil(
    segmap: np.ndarray,
    all_category_ids: Sequence[int],
) -> Image.Image:
    """[summary]

    Args:
        segmap    (np.ndarray; C^(height, width)): a segmentation map
            C: {0, 1, ..., #categories-1}
                0: must be background
        all_category_ids (np.ndarray;  C^(#C, )):

    Returns:
        Image.Image: [description]
    """
    palette = voc_colormap(all_category_ids)

    img = Image.fromarray(segmap.astype(np.uint8)).convert("P")
    img.putpalette(palette.astype(np.uint8))
    return img
