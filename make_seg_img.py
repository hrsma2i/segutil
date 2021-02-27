from pathlib import Path

import typer
import mmcv
import numpy as np
from PIL import Image


def main(ann_dir: Path = "~/data/iccv09Data/labels"):
    # convert dataset annotation to semantic segmentation map
    # define class and plaette for better visualization
    palette = [
        [128, 128, 128],  # sky
        [129, 127, 38],  # tree
        [120, 69, 125],  # road
        [53, 125, 34],  # grass
        [0, 11, 123],  # water
        [118, 20, 12],  # bldg
        [122, 81, 25],  # mntn
        [241, 134, 51],  # fg obj
    ]
    filenames = list(mmcv.scandir(str(ann_dir), suffix=".regions.txt"))
    total = len(filenames)
    for i, fn in enumerate(filenames):
        seg_map = np.loadtxt(str(ann_dir / fn)).astype(np.uint8)
        seg_img = Image.fromarray(seg_map).convert("P")
        seg_img.putpalette(np.array(palette, dtype=np.uint8))
        fn_out = fn.replace(".regions.txt", ".png")
        seg_img.save(str(ann_dir / fn_out))
        print(f"{fn} -> {fn_out}. progress={i+1}/{total}")


if __name__ == "__main__":
    typer.run(main)
