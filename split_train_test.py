from pathlib import Path

import mmcv
import typer


def main(
    split_dir: Path = "../data/iccv09Data/splits",
    ann_dir: Path = "../data/iccv09Data/labels",
    test_rate: float = 1 / 5,
):
    mmcv.mkdir_or_exist(str(split_dir))
    filename_list = [
        Path(filename).stem for filename in mmcv.scandir(str(ann_dir), suffix=".png")
    ]

    with (split_dir / "train.txt").open("w") as f:
        # select first 4/5 as train set
        train_length = int(len(filename_list) * (1 - test_rate))
        f.writelines(line + "\n" for line in filename_list[:train_length])

    with (split_dir / "val.txt").open("w") as f:
        # select last 1/5 as train set
        f.writelines(line + "\n" for line in filename_list[train_length:])


if __name__ == "__main__":
    typer.run(main)
