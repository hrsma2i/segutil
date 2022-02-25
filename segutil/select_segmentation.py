from typing import List
from itertools import groupby

from segutil.metrics import iou
from segutil.segmentation import Segmentation


def select_segmentation(
    segs: List[Segmentation],
    area_th: float,
    nms_iou_th: float,
    over_smaller: bool = True,
    include_categories: List[int] = None,
    exclude_categories: List[int] = None,
) -> List[Segmentation]:
    """Select Valid Segmentations

    Parameters
    ----------
    segs : List[Segmentation]
    area_th : float
        The area threshold to cut off segmentatinos
    nms_iou_th : float
        The IoU threshold to select masks, by default 0.5
    over_smaller : bool, optional
        If this is True, the denominator of IoU is the smaller mask's area
        instead of the union, by default True.
    include_categories : List[int], optional
        Included categories, by default None
    exclude_categories : List[int], optional
        Excluded categories, by default None

    Returns
    -------
    List[Segmentation]
    """

    segs = category_restrict(
        segs,
        include=include_categories,
        exclude=exclude_categories,
    )
    segs = remove_small_segs(segs, th=area_th)
    segs = drop_duplicated_category(segs)
    segs = non_maximum_suppression(
        segs,
        th=nms_iou_th,
        over_smaller=over_smaller,
    )
    return segs


def category_restrict(
    segs: List[Segmentation],
    include: List[int] = None,
    exclude: List[int] = None,
) -> List[Segmentation]:
    """Restrict segmentations by their category

    Parameters
    ----------
    segs : List[Segmentation]
    include : List[int], optional
        Included categories, by default None
    exclude : List[int], optional
        Excluded categories, by default None

    Returns
    -------
    List[Segmentation]
    """
    if include is not None:
        return [s for s in segs if s.category_id in include]
    elif exclude is not None:
        return [s for s in segs if s.category_id not in exclude]
    else:
        ValueError("Give either `include` of `exclude`, only one of them.")


def remove_small_segs(segs: List[Segmentation], th: float) -> List[Segmentation]:
    """Remove segmentations with small area

    Parameters
    ----------
    segs : List[Segmentation]
    th : float
        The area threshold to cut off segmentatinos

    Returns
    -------
    List[Segmentation]
    """
    return [s for s in segs if s.area_rate > th]


def drop_duplicated_category(
    segs: List[Segmentation], except_for: List[int] = []
) -> List[Segmentation]:
    """Drop segmentations with duplicated categories,
    keeping a single segmentation with the heighest score in each category.

    Parameters
    ----------
    segs : List[Segmentation]
    except_for : List[int], optional
        Excluded categories.
        For these categories, duplications will be remained.

    Returns
    -------
    List[Segmentation]
    """
    targets = [s for s in segs if s.category_id not in except_for]
    not_targets = [s for s in segs if s.category_id in except_for]
    groups = groupby(
        sorted(targets, key=lambda s: s.category_id), key=lambda s: s.category_id
    )
    return not_targets + [max(g, key=lambda s: s.score) for _, g in groups]


def non_maximum_suppression(
    segs: List[Segmentation],
    th: float,
    over_smaller: bool = True,
) -> List[Segmentation]:
    """Non Maxismum Supression
    This is a way to select segmentation masks or bounding boxes.
    https://ohke.hateblo.jp/entry/2020/06/20/230000

    Parameters
    ----------
    segs : List[Segmentation]
    th : float, optional
        The IoU threshold to select masks, by default 0.5
    over_smaller : bool, optional
        If this is True, the denominator of IoU is the smaller mask's area
        instead of the union, by default True.

    Returns
    -------
    List[Segmentation]
    """
    if segs == []:
        return []

    max_seg = max(segs, key=lambda s: s.score)
    remainings = [
        s for s in segs if iou(max_seg.mask, s.mask, over_smaller=over_smaller) < th
    ]

    return [max_seg] + non_maximum_suppression(remainings, th=th)
