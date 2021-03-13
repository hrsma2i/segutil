import os.path as osp

import typer
import mmcv
from mmcv import Config
from mmseg.apis import set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from mmseg.models import build_segmentor

classes = ("sky", "tree", "road", "grass", "water", "bldg", "mntn", "fg obj")
palette = [
    [128, 128, 128],
    [129, 127, 38],
    [120, 69, 125],
    [53, 125, 34],
    [0, 11, 123],
    [118, 20, 12],
    [122, 81, 25],
    [241, 134, 51],
]
img_dir = "images"
ann_dir = "labels"
data_root = "../data/iccv09Data"
# config_file = "configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py"
config_file = "configs/fastscnn/fast_scnn_4x8_80k_lr0.12_cityscapes.py"


@DATASETS.register_module()
class StandfordBackgroundDataset(CustomDataset):
    CLASSES = classes
    PALETTE = palette

    def __init__(self, split, **kwargs):
        super().__init__(
            img_suffix=".jpg", seg_map_suffix=".png", split=split, **kwargs
        )
        assert osp.exists(self.img_dir) and self.split is not None


def main(batch_size: int = 2):
    cfg = Config.fromfile(config_file)
    # Since we use ony one GPU, BN is used instead of SyncBN
    cfg.norm_cfg = dict(type="BN", requires_grad=True)
    cfg.model.backbone.norm_cfg = cfg.norm_cfg
    cfg.model.decode_head.norm_cfg = cfg.norm_cfg
    # cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg
    cfg.model.auxiliary_head[0].norm_cfg = cfg.norm_cfg
    cfg.model.auxiliary_head[1].norm_cfg = cfg.norm_cfg

    # modify num classes of the model in decode/auxiliary head
    cfg.model.decode_head.num_classes = len(classes)
    # cfg.model.auxiliary_head.num_classes = 8
    cfg.model.auxiliary_head[0].num_classes = len(classes)
    cfg.model.auxiliary_head[1].num_classes = len(classes)

    cfg.model.decode_head.loss_decode.use_sigmoid = False
    cfg.model.auxiliary_head[0].loss_decode.use_sigmoid = False
    cfg.model.auxiliary_head[1].loss_decode.use_sigmoid = False

    # Modify dataset type and path
    cfg.dataset_type = "StandfordBackgroundDataset"
    cfg.data_root = data_root

    cfg.data.samples_per_gpu = batch_size
    cfg.data.workers_per_gpu = 8

    cfg.img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
    )
    cfg.crop_size = (256, 256)
    cfg.train_pipeline = [
        dict(type="LoadImageFromFile"),
        dict(type="LoadAnnotations"),
        dict(type="Resize", img_scale=(320, 240), ratio_range=(0.5, 2.0)),
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
    cfg.data.train.pipeline = cfg.train_pipeline
    cfg.data.train.split = "splits/train.txt"

    cfg.data.val.type = cfg.dataset_type
    cfg.data.val.data_root = cfg.data_root
    cfg.data.val.img_dir = img_dir
    cfg.data.val.ann_dir = ann_dir
    cfg.data.val.pipeline = cfg.test_pipeline
    cfg.data.val.split = "splits/val.txt"

    cfg.data.test.type = cfg.dataset_type
    cfg.data.test.data_root = cfg.data_root
    cfg.data.test.img_dir = img_dir
    cfg.data.test.ann_dir = ann_dir
    cfg.data.test.pipeline = cfg.test_pipeline
    cfg.data.test.split = "splits/val.txt"

    # We can still use the pre-trained Mask RCNN model though we do not need to
    # use the mask branch
    # cfg.load_from = (
    #     "checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth"
    # )

    # Set up working dir to save files and logs.
    cfg.work_dir = "./work_dirs/tutorial"

    cfg.total_iters = 200
    cfg.log_config.interval = 10
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
