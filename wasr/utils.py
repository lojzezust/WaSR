import os
import torch
from torch import nn
import pytorch_lightning as pl

from collections import OrderedDict
from collections.abc import Iterable, Mapping
from typing import Dict

def load_weights(path):
    state_dict = torch.load(path, map_location='cpu')
    if 'model' in state_dict:
        # Loading weights from checkpoint
        state_dict = state_dict['model']

    return state_dict

class ModelExporter(pl.Callback):
    """Exports model weights at the end of the training."""
    def on_fit_end(self, trainer, pl_module):
        export_path = os.path.join(trainer.log_dir, 'weights.pth')
        torch.save(pl_module.model.state_dict(), export_path)

class IntermediateLayerGetter(nn.ModuleDict):
    """
    Taken from: https://github.com/pytorch/vision/blob/master/torchvision/models/_utils.py

    Module wrapper that returns intermediate layers from a model
    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.
    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.
    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    Examples::
        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


def tensor_map(obj, fn):
    """Map a function to a nested tensor structure.
    Example:
    >>> fn = lambda t: t.to('cpu')
    >>> a = tensor_map(a, fn)
    """

    if torch.is_tensor(obj):
        return fn(obj)

    elif isinstance(obj, Mapping):
        dtype = type(obj)
        res = ((k, tensor_map(v, fn)) for k, v in obj.items())
        res = dtype(res)
        return res
    elif isinstance(obj, Iterable):
        dtype = type(obj)
        res = (tensor_map(v, fn) for v in obj)
        res = dtype(res)
        return res
    else:
        raise TypeError("Invalid type for tensor_map")

