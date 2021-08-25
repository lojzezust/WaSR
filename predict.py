import argparse
import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets.mastr import MaSTr1325Dataset
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
MODEL = 'wasr_resnet101_imu'


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="WaSR Network MaSTr1325 Inference")
    parser.add_argument("--dataset_config", type=str, required=True,
                        help="Path to the file containing the MaSTr1325 dataset mapping.")
    parser.add_argument("--model", type=str, choices=models.model_list, default=MODEL,
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


def predict(args):
    dataset = MaSTr1325Dataset(args.dataset_config, normalize_t=PytorchHubNormalization())
    dl = DataLoader(dataset, batch_size=args.batch_size, num_workers=1)

    # Prepare model
    model = models.get_model(args.model, pretrained=False)
    state_dict = load_weights(args.weights)
    model.load_state_dict(state_dict)
    predictor = Predictor(model, args.fp16)

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    for features, labels in tqdm(iter(dl), total=len(dl)):
        pred_masks = predictor.predict_batch(features)

        for i, pred_mask in enumerate(pred_masks):
            pred_mask = SEGMENTATION_COLORS[pred_mask]
            mask_img = Image.fromarray(pred_mask)

            out_file = output_dir / labels['mask_filename'][i]

            mask_img.save(out_file)

def main():
    args = get_arguments()
    print(args)

    predict(args)


if __name__ == '__main__':
    main()
