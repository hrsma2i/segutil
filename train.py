from functools import reduce
import os.path as osp
from pathlib import Path

import typer
import numpy as np
import mmcv
from mmcv import Config
from mmseg.apis import set_random_seed
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor

# to register the sampler
import segmentation.area_based_sampler
from segmentation.balanced_batch.batch_samplers import BalancedBatchSampler
from segmentation.balanced_batch.train import train_segmentor

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
    class_weight_txt=typer.Option(
        None, help="The relative path for a txt file, where lists classes' weight."
    ),
    area_based_sample: bool = typer.Option(
        False,
        help="Weighten pixels to calculate losses by its class's area inverse.",
    ),
    lovasz: bool = typer.Option(
        False,
        help="Use Lovasz loss instead of SCE.",
    ),
    per_image: bool = typer.Option(
        False,
        help="Use Lovasz loss instead of SCE.",
    ),
    resume_from=typer.Option(
        None,
        help="The (abusolute) path for a checkpoint file to resume.",
    ),
    train_split=typer.Option(
        None, help="The relative path for a txt file listing image names to train on."
    ),
    val_split=typer.Option(
        None,
        help="The relative path for a txt file listing image names to validate and test on.",
    ),
    index_class_csv=typer.Option(
        None,
        help="The relative path for a CSV file includes two columns"
        " used in BalancedBatchSampler; index, class_id.",
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

    if class_weight_txt is not None:
        with (Path(data_root) / class_weight_txt).open() as f:
            class_weight = [float(x.rstrip()) for x in f]

    if not lovasz:
        cfg.model.decode_head.loss_decode.use_sigmoid = False
        cfg.model.auxiliary_head[0].loss_decode.use_sigmoid = False
        cfg.model.auxiliary_head[1].loss_decode.use_sigmoid = False
        if class_weight_txt is not None:
            cfg.model.decode_head.loss_decode.class_weight = class_weight
            cfg.model.auxiliary_head[0].loss_decode.class_weight = class_weight
            cfg.model.auxiliary_head[1].loss_decode.class_weight = class_weight
    else:
        cfg.model.decode_head.loss_decode = dict(
            type="LovaszLoss",
            per_image=per_image,
            reduction="mean" if per_image else "none",
            loss_weight=0.4,
            class_weight=class_weight if class_weight_txt is not None else None,
        )
        cfg.model.auxiliary_head[0].loss_decode = dict(
            type="LovaszLoss",
            per_image=per_image,
            reduction="mean" if per_image else "none",
            loss_weight=0.4,
            class_weight=class_weight if class_weight_txt is not None else None,
        )
        cfg.model.auxiliary_head[1].loss_decode = dict(
            type="LovaszLoss",
            per_image=per_image,
            reduction="mean" if per_image else "none",
            loss_weight=0.4,
            class_weight=class_weight if class_weight_txt is not None else None,
        )

    if area_based_sample:
        cfg.model.decode_head.sampler = dict(type="AreaBasedSampler")
        cfg.model.auxiliary_head[0].sampler = dict(type="AreaBasedSampler")
        cfg.model.auxiliary_head[1].sampler = dict(type="AreaBasedSampler")

    # Modify dataset type and path
    cfg.dataset_type = "CustomDataset"
    cfg.data_root = data_root

    if index_class_csv is None:
        batch_sampler = None
        cfg.data.samples_per_gpu = batch_size
    else:
        batch_sampler = BalancedBatchSampler(osp.join(data_root, index_class_csv))
        cfg.data.samples_per_gpu = batch_sampler.num_classes
    cfg.data.workers_per_gpu = 8

    cfg.img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
    )
    # height, width
    # c.f., https://github.com/open-mmlab/mmsegmentation/issues/30
    cfg.crop_size = (450, 300)
    cfg.train_pipeline = [
        dict(type="LoadImageFromFile"),
        dict(type="LoadAnnotations"),
        dict(type="Resize", img_scale=(900, 600), ratio_range=(0.75, 1.5)),
        # for cat_max_ratio meaning
        # c.f., https://github.com/open-mmlab/mmsegmentation/issues/30
        dict(type="RandomCrop", crop_size=cfg.crop_size, cat_max_ratio=0.5),
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

    if resume_from:
        print(f"resume from {resume_from}")
        cfg.resume_from = resume_from

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
        model,
        datasets,
        cfg,
        distributed=False,
        validate=validate,
        meta=dict(),
        batch_sampler=batch_sampler,
    )


if __name__ == "__main__":
    typer.run(main)
