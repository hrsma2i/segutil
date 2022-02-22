from typing import List, NewType

import numpy as np
from typing_extensions import TypedDict

# COCO Annotation format:
# https://qiita.com/harmegiddo/items/da131ae5bcddbbbde41f

Polygon = NewType("Polygon", List[int])
# A single polygon is 2-D coordinates
# surrounding a single connected mask:
#   [x1, y1, x2, y2, ...]


ImageHeight = NewType("Height", int)
ImageWidth = NewType("Width", int)


class COCORLE(TypedDict):
    size: tuple[ImageHeight, ImageWidth]
    counts: str
