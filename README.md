# Semantic Segmentation Tools

<!-- TOC -->

- [Semantic Segmentation Tools](#semantic-segmentation-tools)
    - [Setup](#setup)
    - [Transform Annotations](#transform-annotations)
        - [COCO RLE → numpy.ndarray](#coco-rle-→-numpyndarray)
    - [Visualize Segmentation Map](#visualize-segmentation-map)

<!-- /TOC -->

## Setup

Install Python libraries via Poetry by the following commands:

```sh
pip install poetry
poetry install -E $EXTRAS
poetry shell
```

You must choose `$EXTRAS` from the following options:

- `fastscnn`: Fast-SCNN dependencies
- `vis`: Visualization dependencies


Install `pycocotools` by the following because Poetry doesn't support git subfolder installation.

```sh
pip install Cython
pip install git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI
```

## Transform Annotations

### COCO RLE → numpy.ndarray

```py
from segmentation.transforms import coco_rle_to_mask
from segmentation.transforms import masks_to_segmap

rle = {
    "size": [
        height #int,
        width #int,
    ],
    "counts": RLE #str like "12345 5 23456 7 456456 9 ...",
}

mask = coco_rle_to_mask(rle)
# mask: np.ndarray in {0, 1}^(height, width)
# One binary mask means a specific class.

# masks: a list of binary masks for several categories in a single image.
# class_ids: a list of class ids for each binary mask.
segmap = masks_to_segmap(masks, class_ids)
# segmap: np.ndarray in {0, 1, ..., #classes}^(height, width). 0 is background.
```

## Visualize Segmentation Map

Example (Jupyter Notebook):

```py
from segmentation.visualizations import vis_segmap

# img: np.ndarray in {0, 1, ..., 255}^(height, width, RGB)
# segmap: np.ndarray in {0, 1, ..., #classes}^(height, width). 0 is background.

categories = ["background", "car", "bike", ...]
ax, legend_handles = vis_segmap(
    img,
    segmap,
    label_names=categories,
    alpha=0.7,
)
ax.legend(handles=legend_handles, bbox_to_anchor=(1, 1), loc=2)
```
