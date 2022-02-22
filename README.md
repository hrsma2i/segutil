# SegUtil: Semantic Segmentation Utilities

<!-- TOC -->

- [SegUtil: Semantic Segmentation Utilities](#segutil-semantic-segmentation-utilities)
    - [Setup](#setup)
        - [Pip](#pip)
        - [Poetry](#poetry)
        - [For Developer](#for-developer)
        - [Common](#common)
    - [Transform Annotations](#transform-annotations)
        - [COCO RLE → numpy.ndarray](#coco-rle-→-numpyndarray)
    - [Visualize Segmentation Map](#visualize-segmentation-map)

<!-- /TOC -->

## Setup

Install Python libraries via Poetry by the following commands:

### Pip

```sh
pip install git+https://github.com/hrsma2i/segutil.git
```

### Poetry

```sh
poetry add git+https://github.com/hrsma2i/segutil.git#main
```


### For Developer

```sh
git clone https://github.com/hrsma2i/segutil.git
pip install poetry
poetry shell
```

### Common

Install `pycocotools` by the following because Poetry doesn't support git subfolder installation.

```sh
pip install Cython
pip install git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI
```

## Transform Annotations

### COCO RLE → numpy.ndarray

```py
from segutil.transforms import decode_mask
from segutil.transforms import masks_to_segmap

rle = {
    "size": [
        height #int,
        width #int,
    ],
    "counts": RLE #str like "12345 5 23456 7 456456 9 ...",
}

mask = decode_mask(rle)
# mask: np.ndarray in {0,1 (np.int32)}^(height, width)
# One binary mask means a specific category.

# masks: a list of binary masks for several categories in a single image.
# category_ids: a list of category ids for each binary mask.
segmap = masks_to_segmap(masks, category_ids)
# segmap: np.ndarray in {0, 1, ..., #categories}^(height, width). 0 is background.
```

## Visualize Segmentation Map

Example (Jupyter Notebook):

```py
from segutil.visualizations import vis_segmap

# img: np.ndarray in {0, 1, ..., 255}^(height, width, RGB)
# segmap: np.ndarray in {0, 1, ..., #categories}^(height, width). 0 is background.

categories = ["background", "car", "bike", ...]
vis_segmap(
    img,
    segmap,
    category_names=categories,
    alpha=0.7,
)
```
