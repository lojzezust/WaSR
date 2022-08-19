import argparse
import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from functools import partial
import albumentations as A
import cv2

from datasets.mods import MODSDataset
from datasets.transforms import AlbumentationsTransform, PytorchHubNormalization
from wasr.inference import LitPredictor
import wasr.models as models
from wasr.utils import load_weights


# Colors for each class. Should correspond to the colors specified in the MODS evaluator config.
SEGMENTATION_COLORS = np.array([
    [0, 0, 0],    # [247, 195,  37] # Obstacle
    [255, 0, 0],  # [ 41, 167, 224] # Water
    [0, 255, 0],  # [ 90,  75, 164] # Sky
], np.uint8)

SIZE = (512,384)
BATCH_SIZE = 4
WORKERS = 4
DATASET_PATH = os.path.expanduser('~/data/datasets/mods')
ARCHITECTURE = 'wasr_resnet101_imu'


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="WaSR Network MODS Inference")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Minibatch size (number of samples) on each GPU.")
    parser.add_argument("--workers", type=int, default=WORKERS,
                        help="Number of dataloader workers (per GPU).")
    parser.add_argument("--dataset-path", type=str, default=DATASET_PATH,
                        help="Path to the MODS dataset root.")
    parser.add_argument("--architecture", type=str, choices=models.model_list, default=ARCHITECTURE,
                        help="Model architecture.")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to the model weights or a model checkpoint.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory for output prediction saving.")
    parser.add_argument("--fp16", action='store_true',
                        help="Use half precision for inference.")
    parser.add_argument("--gpus", default=-1,
                    help="Number of gpus (or GPU ids) used for training (used by the distributed code, don't set manually).")
    return parser.parse_args()

def export_predictions(probs, batch, output_dir=None):
    features, metadata = batch

    # Class prediction
    out_class = probs.argmax(1).astype(np.uint8)

    for i, pred_mask in enumerate(out_class):
            pred_mask = SEGMENTATION_COLORS[pred_mask]
            mask_img = Image.fromarray(pred_mask)

            seq_dir = output_dir / metadata['seq'][i]
            if not seq_dir.exists():
                seq_dir.mkdir(parents=True, exist_ok=True)

            out_file = (seq_dir / metadata['name'][i]).with_suffix('.png')
            mask_img.save(out_file)

def predict_mods(args):

    # Resize to images WaSR size
    transform = AlbumentationsTransform(A.Resize(SIZE[1], SIZE[0], interpolation=cv2.INTER_AREA))

    # Create augmentation transform if not disabled
    dataset = MODSDataset(args.dataset_path, transform=transform, normalize_t=PytorchHubNormalization())
    dl = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)

    # Load model
    model = models.get_model(args.architecture, pretrained=False)
    state_dict = load_weights(args.weights)
    model.load_state_dict(state_dict)

    output_dir = Path(args.output_dir)

    export_fn = partial(export_predictions, output_dir=output_dir)
    predictor = LitPredictor(model, export_fn)

    precision = 16 if args.fp16 else 32
    trainer = pl.Trainer(gpus=args.gpus,
                         strategy='ddp',
                         precision=precision,
                         logger=False)

    trainer.predict(predictor, dl)

def main():
    args = get_arguments()
    print(args)

    predict_mods(args)

if __name__ == '__main__':
    main()
