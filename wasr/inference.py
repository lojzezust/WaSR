from PIL import Image

import torch
import torchvision.transforms.functional as TF

from .utils import tensor_map
import warnings

class Predictor():
    def __init__(self, model, half_precision=False):
        self.model = model
        self.half_precision = half_precision

        use_gpu = torch.cuda.is_available()
        self.device = torch.device('cuda:0') if use_gpu else torch.device('cpu')

        if self.half_precision:
            self.model = self.model.half()
        self.model = self.model.eval().to(self.device)

    def predict_batch(self, batch):

        map_fn = lambda t: t.to(self.device)
        batch = tensor_map(batch, map_fn)

        with torch.no_grad():
            if self.half_precision:
                with torch.cuda.amp.autocast():
                    res = self.model(batch)
            else:
                res = self.model(batch)

        out = res['out'].cpu().detach()

        size = (batch['image'].size(2), batch['image'].size(3))
        out = TF.resize(out, size, interpolation=Image.BILINEAR)
        out_class = out.argmax(1)
        out_class = out_class.byte().numpy()

        return out_class


try:
    from pytorch_lightning import LightningModule
except:
    warnings.warn('PyTorch Lightning is required for some features (training, distributed inference).')
    LightningModule = None

class LitPredictor(LightningModule):
    """Lightning model wrapper for running inference. Supports multi-gpu inference."""
    def __init__(self, model, export_fn, raw=False):
        super().__init__()
        self.model = model
        self.export_fn = export_fn
        self.raw = raw

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        features, metadata = batch
        outputs = self.model(features)
        if self.raw:
            # Keep raw input and device (e.g. for mask filling)
            self.export_fn(outputs, batch)
            return

        out = outputs['out'].cpu().detach()

        # Upscale
        size = (features['image'].size(2), features['image'].size(3))
        out = TF.resize(out, size, interpolation=Image.BILINEAR)
        out = out.numpy()

        self.export_fn(out, batch)
