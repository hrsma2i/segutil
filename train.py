import os.path as osp
from pathlib import Path

import typer
import numpy as np
import mmcv
from mmcv import Config
from mmseg.apis import set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor

config_file = "configs/fastscnn/fast_scnn_4x8_80k_lr0.12_cityscapes.py"


def main(
    data_root=typer.Option(
        ..., help="The starting point for the following relative paths."
    ),
    img_dir=typer.Option(
        ...,
        help="The relative path for input JPGs.",
    ),
    ann_dir=typer.Option(
        ...,
        help="The relative path for annotatoin PNGs.",
    ),
    classes_txt=typer.Option(
        ...,
        help="The relative path for a txt file, which lists classes' name.",
    ),
    palette_txt=typer.Option(
        ...,
        help="The relative path for a txt file, where a np.ndarray is saved,"
        " whose shape is (#classes, RGB)",
    ),
    max_iters: int = 80000,
    log_interval: int = 10,
    batch_size: int = 2,
):
    with (Path(data_root) / classes_txt).open() as f:
        classes = [x for x in f]
    palette = np.loadtxt(Path(data_root) / palette_txt).astype(int).tolist()

    cfg = Config.fromfile(config_file)
    # Since we use ony one GPU, BN is used instead of SyncBN
    cfg.norm_cfg = dict(type="BN", requires_grad=True)
    cfg.model.backbone.norm_cfg = cfg.norm_cfg
    cfg.model.decode_head.norm_cfg = cfg.norm_cfg
    cfg.model.auxiliary_head[0].norm_cfg = cfg.norm_cfg
    cfg.model.auxiliary_head[1].norm_cfg = cfg.norm_cfg

    # modify num classes of the model in decode/auxiliary head
    cfg.model.decode_head.num_classes = len(classes)
    cfg.model.auxiliary_head[0].num_classes = len(classes)
    cfg.model.auxiliary_head[1].num_classes = len(classes)

    cfg.model.decode_head.loss_decode.use_sigmoid = False
    cfg.model.auxiliary_head[0].loss_decode.use_sigmoid = False
    cfg.model.auxiliary_head[1].loss_decode.use_sigmoid = False

    # Modify dataset type and path
    cfg.dataset_type = "CustomDataset"
    cfg.data_root = data_root

    cfg.data.samples_per_gpu = batch_size
    cfg.data.workers_per_gpu = 8

    cfg.img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
    )
    cfg.crop_size = (512, 512)
    cfg.train_pipeline = [
        dict(type="LoadImageFromFile"),
        dict(type="LoadAnnotations"),
        dict(type="Resize", img_scale=(1500, 1000), ratio_range=(0.5, 2.0)),
        dict(type="RandomCrop", crop_size=cfg.crop_size, cat_max_ratio=0.75),
        dict(type="RandomFlip", prob=0.5),
        dict(type="PhotoMetricDistortion"),
        dict(type="Normalize", **cfg.img_norm_cfg),
        dict(type="Pad", size=cfg.crop_size, pad_val=0, seg_pad_val=255),
        dict(type="DefaultFormatBundle"),
        dict(type="Collect", keys=["img", "gt_semantic_seg"]),
    ]

    cfg.test_pipeline = [
        dict(type="LoadImageFromFile"),
        dict(
            type="MultiScaleFlipAug",
            img_scale=(320, 240),
            # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
            flip=False,
            transforms=[
                dict(type="Resize", keep_ratio=True),
                dict(type="RandomFlip"),
                dict(type="Normalize", **cfg.img_norm_cfg),
                dict(type="ImageToTensor", keys=["img"]),
                dict(type="Collect", keys=["img"]),
            ],
        ),
    ]

    cfg.data.train.type = cfg.dataset_type
    cfg.data.train.data_root = cfg.data_root
    cfg.data.train.img_dir = img_dir
    cfg.data.train.ann_dir = ann_dir
    cfg.data.train.palette = palette
    cfg.data.train.classes = classes
    cfg.data.train.pipeline = cfg.train_pipeline
    cfg.data.train.split = None  # "splits/train.txt"

    cfg.data.val.type = cfg.dataset_type
    cfg.data.val.data_root = cfg.data_root
    cfg.data.val.img_dir = img_dir
    cfg.data.val.ann_dir = ann_dir
    cfg.data.val.palette = palette
    cfg.data.val.classes = classes
    cfg.data.val.pipeline = cfg.test_pipeline
    cfg.data.val.split = None  # "splits/val.txt"

    cfg.data.test.type = cfg.dataset_type
    cfg.data.test.data_root = cfg.data_root
    cfg.data.test.img_dir = img_dir
    cfg.data.test.ann_dir = ann_dir
    cfg.data.test.palette = palette
    cfg.data.test.classes = classes
    cfg.data.test.pipeline = cfg.test_pipeline
    cfg.data.test.split = None  # "splits/val.txt"

    # We can still use the pre-trained Mask RCNN model though we do not need to
    # use the mask branch
    # cfg.load_from = (
    #     "checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth"
    # )

    # Set up working dir to save files and logs.
    cfg.work_dir = "./work_dirs/tutorial"

    cfg.runner.max_iters = max_iters
    cfg.log_config.interval = log_interval
    cfg.evaluation.interval = 200
    cfg.checkpoint_config.interval = 200

    # Set seed to facitate reproducing the result
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)

    # Let's have a look at the final config used for training
    print(f"Config:\n{cfg.pretty_text}")

    # Build the dataset
    datasets = [build_dataset(cfg.data.train)]

    # Build the detector
    model = build_segmentor(cfg.model)
    # Add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    # Create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    train_segmentor(model, datasets, cfg, distributed=False, validate=True, meta=dict())


if __name__ == "__main__":
    typer.run(main)
