import argparse
import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets.folder import FolderDataset
from datasets.transforms import PytorchHubNormalization
from wasr.inference import Predictor
import wasr.models as models
from wasr.utils import load_weights


# Colors corresponding to each segmentation class
SEGMENTATION_COLORS = np.array([
    [247, 195, 37],
    [41, 167, 224],
    [90, 75, 164]
], np.uint8)

BATCH_SIZE = 12
ARCHITECTURE = 'wasr_resnet101_imu'


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="WaSR Network MaSTr1325 Inference")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Path to the directory containing input images.")
    parser.add_argument("--imu_dir", type=str, default=None,
                        help="(optional) Path to the directory containing input IMU masks.")
    parser.add_argument("--architecture", type=str, choices=models.model_list, default=ARCHITECTURE,
                        help="Model architecture.")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to the model weights or a model checkpoint.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Minibatch size (number of samples) used on each device.")
    parser.add_argument("--fp16", action='store_true',
                        help="Use half precision for inference.")
    return parser.parse_args()

def export_predictions(preds, batch, output_dir):
    features, metadata = batch

    for i, pred_mask in enumerate(preds):
        pred_mask = SEGMENTATION_COLORS[pred_mask]
        mask_img = Image.fromarray(pred_mask)

        out_path = output_dir / Path(metadata['image_path'][i]).with_suffix('.png')
        out_path.parent.mkdir(parents=True, exist_ok=True)

        mask_img.save(str(out_path))

def predict(args):
    dataset = FolderDataset(args.image_dir, args.imu_dir, normalize_t=PytorchHubNormalization())
    dl = DataLoader(dataset, batch_size=args.batch_size, num_workers=1)

    # Prepare model
    model = models.get_model(args.architecture, pretrained=False)
    state_dict = load_weights(args.weights)
    model.load_state_dict(state_dict)
    predictor = Predictor(model, args.fp16)

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    for batch in tqdm(iter(dl), total=len(dl)):
        features, _ = batch
        pred_masks = predictor.predict_batch(features)

        export_predictions(pred_masks, batch, output_dir)

def main():
    args = get_arguments()
    print(args)

    predict(args)


if __name__ == '__main__':
    main()
