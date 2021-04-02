from pathlib import Path

import pandas as pd
from torch.utils.data.sampler import Sampler, RandomSampler


class BalancedBatchSampler(Sampler):
    def __init__(self, index_class_csv: Path):
        """

        Args:
            index_class_csv (Path):
                index,class_id
                    1,       2
                    2,       5
                    3,       3
        """
        groups = pd.read_csv(index_class_csv).groupby("class_id")

        self.sampler_infos = {}
        for c, g in groups["index"]:
            indices = g.tolist()
            self.sampler_infos[c] = {
                "sampler_iter": iter(RandomSampler(indices)),
                "indices": indices,
                "is_longest": False,
            }

        self.max_len = max(
            [len(info["indices"]) for info in self.sampler_infos.values()]
        )

        for info in self.sampler_infos.values():
            if len(info["indices"]) == self.max_len:
                info["is_longest"] = True

    @property
    def num_classes(self):
        return len(self.sampler_infos)

    def __len__(self):
        return self.max_len

    def __iter__(self):
        while True:
            batch = list()
            for info in self.sampler_infos.values():
                try:
                    ii = next(info["sampler_iter"])
                    i = info["indices"][ii]
                    batch.append(i)
                except StopIteration:
                    if info["is_longest"]:
                        raise StopIteration
                    else:
                        info["sampler_iter"] = iter(RandomSampler(info["indices"]))
                        ii = next(info["sampler_iter"])
                        i = info["indices"][ii]
                        batch.append(i)
            yield batch
