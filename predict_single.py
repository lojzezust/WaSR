import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch

import wasr.models as models

NORM_MEAN = np.array([0.485, 0.456, 0.406])
NORM_STD = np.array([0.229, 0.224, 0.225])

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
    parser.add_argument("image", type=str,
                        help="Path to the image to run inference on.")
    parser.add_argument("output", type=str,
                        help="Path to the file, where the output prediction will be saved.")
    parser.add_argument("--imu_mask", type=str, default=None,
                        help="Path to the corresponding IMU mask (if needed by the model).")
    parser.add_argument("--architecture", type=str, choices=models.model_list, default=ARCHITECTURE,
                        help="Model architecture.")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to the model weights or a model checkpoint.")
    return parser.parse_args()


def predict_image(model, image, imu_mask=None):
    feat = {'image': image.cuda()}
    if imu_mask is not None:
        feat['imu_mask'] = imu_mask.cuda()

    res = model(feat)
    prediction = res['out'].detach().softmax(1).cpu()
    return prediction


def predict(args):
    # Load and prepare model
    model = models.get_model(args.architecture, pretrained=False)

    state_dict = torch.load(args.weights, map_location='cpu')
    if 'model' in state_dict:
        # Loading weights from checkpoint
        state_dict = state_dict['model']
    model.load_state_dict(state_dict)

    # Enable eval mode and move to CUDA
    model = model.eval().cuda()

    # Load and normalize image
    img = np.array(Image.open(args.image))
    H,W,_ = img.shape
    img = torch.from_numpy(img) / 255.0
    img = (img - NORM_MEAN) / NORM_STD
    img = img.permute(2,0,1).unsqueeze(0) # [1xCxHxW]
    img = img.float()

    # Load IMU mask if provided
    imu_mask = None
    if args.imu_mask is not None:
        imu_mask = np.array(Image.open(args.imu_mask))
        imu_mask = imu_mask.astype(np.bool)
        imu_mask = torch.from_numpy(imu_mask).unsqueeze(0) # [1xHxW]


    # Run inference
    probs = predict_image(model, img, imu_mask)
    probs = torch.nn.functional.interpolate(probs, (H,W), mode='bilinear')
    preds = probs.argmax(1)[0]

    # Convert predictions to RGB class colors
    preds_rgb = SEGMENTATION_COLORS[preds]
    preds_img = Image.fromarray(preds_rgb)

    output_dir = Path(args.output).parent
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    preds_img.save(args.output)

def main():
    args = get_arguments()
    print(args)

    predict(args)


if __name__ == '__main__':
    main()
