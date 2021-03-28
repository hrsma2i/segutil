import torch
from mmseg.core.seg.sampler import BasePixelSampler
from mmseg.core.seg.builder import PIXEL_SAMPLERS

# avoid division by 0
EPS = torch.finfo(torch.float32).eps


@PIXEL_SAMPLERS.register_module()
class AreaBasedSampler(BasePixelSampler):
    def __init__(self, context):
        super().__init__()
        self.context = context

    def sample(self, seg_logit, seg_label):
        """Sample class_areas that have high loss or with low prediction confidence.
        Args:
            seg_logit (torch.Tensor): segmentation logits, shape (N, C, H, W)
            seg_label (torch.Tensor): segmentation label, shape (N, 1, H, W)
        Returns:
            torch.Tensor: segmentation weight, shape (N, H, W)
        """
        with torch.no_grad():
            batch_size, num_all_classes, _, _ = seg_logit.shape
            masks = torch.cat(
                [(seg_label == i).float() for i in range(num_all_classes)], axis=1
            )
            # masks: (N, C, H, W)

            # Normalize so that the class_areas for the class whose area is the smallest, is 1.
            class_areas = torch.sum(masks, dim=(2, 3))
            # class_areas: (N, C)

            class_weights = torch.max(class_areas) / (EPS + class_areas)
            # class_weights: (N, C)

            weighted_masks = (
                masks.permute(2, 3, 0, 1)
                * class_weights.reshape((1, 1, batch_size, num_all_classes))
            ).permute(2, 3, 0, 1)
            # weighted_masks: (N, C, H, W)

            seg_weight = torch.sum(weighted_masks, dim=1)
            # seg_weight: (N, H, W)
            return seg_weight
