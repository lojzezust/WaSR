# WaSR: A Water Segmentation and Refinement Maritime Obstacle Detection Network

PyTorch re-implementation of the WaSR network [[1](#ref1)]. Contains training code, prediction code and models pretrained on the MaSTr1325 dataset[[2](#ref2)].

## 1. Setup

Install the requirements provided in `requirements.txt`.

```bash
pip install -r requirements.txt
```
## 2. Training

1. Download and prepare the [MaSTr1325 dataset](https://box.vicos.si/borja/viamaro/index.html#mastr1325) (images and GT masks). If you plan to use the IMU-enabled model also download the IMU masks.
2. Edit the dataset configuration (`configs/mastr1325_train.yaml`, `configs/mastr1325_val.yaml`) files so that they correctly point to the prepared directories.
3. Use the `train.py` to train the network.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3 # GPUs to use
python train.py \
--train_config configs/mastr1325_train.yaml \
--val_config configs/mastr1325_val.yaml \
--model_name my_wasr \ 
--validation \
--batch_size 4 \
--epochs 50
```

### 2.1 Logging and model weights

A log dir with the specified model name will be created inside the `output` directory. Model checkpoints and training logs will be stored here. At the end of the training the model weights are also exported to a `weights.pth` file inside this directory.

### 2.2 Model architectures

By default the ResNet-101, IMU-enabled version of the WaSR is used in training. To select a different model architecture use the `--model` flag. Currently available model architectures:

| model              | backbone   | IMU |
|--------------------|------------|-----|
| wasr_resnet101_imu | ResNet-101 | ✓   |
| wasr_resnet101     | ResNet-101 |     |
| wasr_resnet50_imu  | ResNet-50  | ✓   |
| wasr_resnet50      | ResNet-50  |     |
| deeplab            | ResNet-101 |     |

## Prediction

##