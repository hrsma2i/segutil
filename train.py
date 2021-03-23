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
    train_split=typer.Option(
        None, help="The relative path for a txt file listing image names to train on."
    ),
    val_split=typer.Option(
        None,
        help="The relative path for a txt file listing image names to validate and test on.",
    ),
    out_dir=typer.Option(
        ...,
        help="The (abusolute) path for a directory where model weights will be saved.",
    ),
    validate: bool = False,
    max_iters: int = 80000,
    log_interval: int = 1,
    eval_interval: int = 1000,
    save_interval: int = 1000,
    batch_size: int = 30,
):
    with (Path(data_root) / classes_txt).open() as f:
        classes = [x.rstrip() for x in f]
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
    cfg.data.train.split = train_split

    cfg.data.val.type = cfg.dataset_type
    cfg.data.val.data_root = cfg.data_root
    cfg.data.val.img_dir = img_dir
    cfg.data.val.ann_dir = ann_dir
    cfg.data.val.palette = palette
    cfg.data.val.classes = classes
    cfg.data.val.pipeline = cfg.test_pipeline
    cfg.data.val.split = val_split

    # We can still use the pre-trained Mask RCNN model though we do not need to
    # use the mask branch
    # cfg.load_from = (
    #     "checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth"
    # )

    # Set up working dir to save files and logs.
    cfg.work_dir = out_dir

    cfg.runner.max_iters = max_iters
    cfg.log_config.interval = log_interval
    cfg.evaluation.interval = eval_interval
    cfg.checkpoint_config.interval = save_interval
    cfg.checkpoint_config.meta = dict(
        CLASSES=classes,
        PALETTE=palette,
    )

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
    cfg.dump(osp.join(out_dir, "config.py"))
    train_segmentor(
        model, datasets, cfg, distributed=False, validate=validate, meta=dict()
    )


if __name__ == "__main__":
    typer.run(main)
