# Semantic Segmentation Tools

<!-- TOC -->

- [Semantic Segmentation Tools](#semantic-segmentation-tools)
    - [Setup](#setup)

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
