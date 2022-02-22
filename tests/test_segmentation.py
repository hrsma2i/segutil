import json

import pytest

from segutil.segmentaion import Segmentation, Bbox
from segutil.transforms import encode_mask
import numpy as np


@pytest.fixture
def mask():
    m = np.zeros((300, 600))
    m[:150, :300] = 1
    return m


@pytest.fixture
def seg(mask: np.ndarray):
    h, w = mask.shape
    return Segmentation(
        filename="sample.jpg",
        height=h,
        width=w,
        encoded_mask=encode_mask(mask),
        category_id=0,
        bbox=Bbox(10, 20, 30, 40),
        score=0.2,
    )


def test_mask_property(mask: np.ndarray, seg: Segmentation):
    assert np.all(seg.mask == mask)


def test_json_serialize(seg: Segmentation):
    s = json.dumps(seg.to_dict())
    d = json.loads(s)

    d["category_id"] = str(d["category_id"])
    d["score"] = str(d["score"])

    seg = Segmentation.from_dict(d)

    # check forced casting
    assert isinstance(d["category_id"], str)
    assert isinstance(seg.category_id, int)
    assert isinstance(d["score"], str)
    assert isinstance(seg.score, float)


def test_are_rate(seg: Segmentation):
    assert seg.area_rate == 1 / 4
