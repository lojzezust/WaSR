This is the PyTorch re-implementation of the ICRA2020 paper: [(Bovcon et al.) A water-obstacle separation and refinement network for unmanned surface vehicles](https://arxiv.org/abs/2001.01921)

## Requirements

Requirements are provided in `requirements.txt`.

```bash
pip install -r requirements.txt
```

## Training

You can use the `train.py` script for basic training of the WaSR model.

```bash
# export CUDA_VISIBLE_DEVICES=-1 # CPU only
export CUDA_VISIBLE_DEVICES=0,1,2,3 # GPUs to use
python -m tools.train --epochs 5
```

To train the IMU version of the network use the `--imu` flag.
```bash
python -m tools.train --epochs 5 --imu
```
