import torch
from torchmetrics import Metric

class PixelAccuracy(Metric):
    def __init__(self, num_classes, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.num_classes = num_classes

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        self.correct += torch.sum(preds == target)
        self.total += torch.sum(target < self.num_classes) # Ignore not used classes (ignore region)

    def compute(self):
        return self.correct.float() / self.total

class ClassIoU(Metric):
    def __init__(self, class_i, num_classes, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("intersection", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("union", default=torch.tensor(0), dist_reduce_fx="sum")
        self.class_i = class_i
        self.num_classes = num_classes

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        preds_mask = preds == self.class_i
        target_mask = target == self.class_i

        # Ignore predictions in ignore regions
        valid_mask = target < self.num_classes
        preds_mask = preds_mask & valid_mask

        self.intersection += torch.sum(preds_mask & target_mask)
        self.union += torch.sum(preds_mask | target_mask)

    def compute(self):
        return self.intersection.float() / self.union.clip(min=1)
