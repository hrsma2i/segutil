from typing import Sequence, List, Union

import numpy as np
from PIL import Image
from pycocotools import mask as mutils

from segutil.visualizations import voc_colormap
from segutil.types import Polygon, COCORLE, is_rle


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
    elif is_rle(encoded_mask):
        return decode_rle(encoded_mask)
    else:
        raise ValueError(f"invalid encoding type: {type(encoded_mask)}")


def decode_polygon(polygons: List[Polygon], height: int, width: int) -> np.ndarray:
    return np.max(mutils.decode(mutils.frPyObjects(polygons, height, width)), axis=2)


def decode_rle(rle: COCORLE) -> np.ndarray:
    return mutils.decode(rle)


def encode_mask(mask: np.ndarray) -> COCORLE:
    return mutils.encode(np.asfortranarray(mask.astype(np.uint8)))


def bytes_rle_to_str(rle: COCORLE) -> dict:
    h, w = rle["size"]
    return {
        "size": (int(h), int(w)),
        "counts": rle["counts"].decode(),
    }


def str_rle_to_bytes(rle: dict) -> COCORLE:
    h, w = rle["size"]
    return {
        "size": (int(h), int(w)),
        "counts": rle["counts"].encode(),
    }


def masks_to_segmap(masks: List[np.ndarray], category_ids: List[int]) -> np.ndarray:
    """Compress binary masks to a segmentatoin map for a particular image

    Parameters
    ----------
    masks : List[np.ndarray {0,1 (np.int32)}^(height, width)]
        decoded binary masks
        len(masks): the number of masks for a single images
    category_ids : List[int]
        each mask's category
        len(category_ids): the number of masks for a single images
        This ranges in {0, 1, ..., C}.
        0: background
        C: the number of all categories

    Returns
    -------
    np.ndarray {0, 1, ..., C (np.int32)}^(height, width)
        0: background
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
