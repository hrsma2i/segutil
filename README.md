# Semantic Segmentation Tools

<!-- TOC -->

- [Semantic Segmentation Tools](#semantic-segmentation-tools)
    - [Setup](#setup)
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

## Visualize Segmentation Map

Example (Jupyter Notebook):

```py
from segmentation.visualizations import vis_segmap

# img: np.ndarray in {0, 1, ..., 255}^(height, width, RGB)
# segmap: np.ndarray in {category_id | 0, 1, ..., C}^(height, width)

categories = ["background", "car", "bike", ...]
ax, legend_handles = vis_segmap(
    img,
    segmap,
    label_names=categories,
    alpha=0.7,
)
ax.legend(handles=legend_handles, bbox_to_anchor=(1, 1), loc=2)
```
