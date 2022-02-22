from dataclasses import dataclass, asdict, fields, Field
from typing import List, Dict

import numpy as np
from segutil.transforms import decode_mask
from segutil.types import EncodedMask, COCORLE, is_rle


@dataclass(frozen=True)
class Bbox:
    left: int
    top: int
    right: int
    bottom: int

    @classmethod
    def from_coco_array(cls, x: List[float]):
        return cls(left=x[0], top=[1], width=[2], height=[3])


@dataclass(frozen=True)
class Segmentation:
    filename: str
    height: int
    width: int
    encoded_mask: EncodedMask
    category_id: int
    bbox: Bbox
    score: float = None
    # mask area / bbox area
    mask_area_fraction: float = None
    mask_mean_score: float = None

    @property
    def mask(self):
        return decode_mask(
            self.encoded_mask,
            height=self.height,
            width=self.width,
        )

    @property
    def area_rate(self):
        image_area = self.height * self.width
        mask_area = np.sum(self.mask)
        return mask_area / image_area

    def to_dict(self):
        return asdict(
            self,
            dict_factory=lambda items: {
                k: self._make_serializable(k, v) for k, v in items if v is not None
            },
        )

    def _make_serializable(self, k, v):
        if k == "encoded_mask" and is_rle(v):
            counts: bytes = v["counts"]
            return COCORLE(size=v["size"], counts=counts.decode())
        else:
            return v

    @classmethod
    def from_dict(cls, d: Dict) -> "Segmentation":
        return cls(
            **{
                f.name: cls._make_deserializable(f, d[f.name])
                for f in fields(cls)
                if d.get(f.name) is not None
            }
        )

    @classmethod
    def _make_deserializable(cls, f: Field, v):
        k = f.name
        if k == "encoded_mask" and is_rle(v):
            counts: str = v["counts"]
            h, w = v["size"]
            return dict(size=(int(h), int(w)), counts=counts.encode())
        if k == "bbox":
            return Bbox(**v)
        else:
            return f.type(v)
