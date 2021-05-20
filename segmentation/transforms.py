from typing import Sequence, List, Dict, Any, Union

import numpy as np
from PIL import Image
from pycocotools import mask as mutils

from segmentation.visualizations import voc_colormap


def coco_ann_to_mask(
    ann: Union[List[List[int]], Dict[str, Any]],
    height: int = None,
    width: int = None,
) -> np.ndarray:
    """[summary]

    Args:
        ann (Union[polygons, rle]):
            - polygons: List[List[int]]
            - rle (Dict[str, Any]): a dict with the follwing schema;
                {
                    "size": [height(int), width(int))],
                    "counts": RLE(str),
                }
        height (int): This is necessary when ann is a polygon.
        width (int): This is necessary when ann is a polygon.

    Raises:
        ValueError: annotation format is neither polygons or rle.

    Returns:
        np.ndarray; {0,1}^(height, width): a binary mask
    """
    if isinstance(ann, list):
        return coco_polygon_to_mask(ann, height, width)
    elif isinstance(ann, dict) and "counts" in ann.keys():
        return coco_rle_to_mask(ann)
    else:
        raise ValueError(f"annotation is invalid type {type(ann)}")


def coco_polygon_to_mask(
    polygons: List[List[int]], height: int, width: int
) -> np.ndarray:
    """[summary]

    Args:
        polygons (List[List[int]]): [description]
        height (int): [description]
        width (int): [description]

    Returns:
        np.ndarray; {0,1}^(height, width): a binary mask
    """
    return np.max(mutils.decode(mutils.frPyObjects(polygons, height, width)), axis=2)


def coco_rle_to_mask(rle: Dict[str, Any]) -> np.ndarray:
    """[summary]

    Args:
        rle (Dict[str, Any]): a dict with the follwing schema;
            {
                "size": [height(int), width(int))],
                "counts": RLE(str),
            }

    Returns:
        np.ndarray; {0, 1}^(height, width): a binary mask
    """
    return mutils.decode(rle)


def masks_to_segmap(masks: List[np.ndarray], class_ids: List[int]) -> np.ndarray:
    """Compress binary masks to a segmentatoin map for a particular image

    Args:
        masks     (List[np.ndarray]; [{0, 1}^(height, width) * #RLEs]): binary masks
            #RLEs: the number of RLEs for a single image.
        class_ids (List[int]): A list of class ids for each binary mask.
            This ranges in {0, 1, ..., #classes}.
            0: must be background
            #classes: The number of all classes.

    Returns:
        segmap (np.ndarray; C^(height, width)): a segmentation map
    """
    masks = np.array(masks)
    class_ids = np.array(class_ids)
    segmap = np.max(class_ids * masks, axis=0).astype(np.uint8)
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
