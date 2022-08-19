import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF

class MODSDataset(torch.utils.data.Dataset):
    """MODS dataset wrapper

    Args:
        mods_dir (str): Path to directory containing the extracted MODS dataset.
        transform (optional): Transform to apply to image and masks
        normalize_t (optional): a normalization transform applied at the end of the pipeline
    """

    def __init__(self, mods_dir, transform=None, normalize_t=None):
        base_dir = mods_dir
        sequences_dir = os.path.join(base_dir, 'sequences')

        data = []
        sequences = os.listdir(sequences_dir)
        for seq in sorted(sequences):
            frame_dir = os.path.join(sequences_dir, seq, 'frames')
            imu_dir = os.path.join(sequences_dir, seq, 'imus')

            for imu_fn in sorted(os.listdir(imu_dir)):
                frame_name = os.path.splitext(imu_fn)[0]
                frame_fn = frame_name + '.jpg'

                frame_data = {
                    'image_path': os.path.join(frame_dir, frame_fn),
                    'imu_path': os.path.join(imu_dir, imu_fn),
                    'name': frame_fn,
                    'seq': seq
                }

                data.append(frame_data)

        self.data = data
        self.transform = transform
        self.normalize_t = normalize_t

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        entry = self.data[idx]

        img = np.array(Image.open(entry['image_path']))
        imu = np.array(Image.open(entry['imu_path']))

        data = {
            'image': img,
            'imu_mask': imu
        }

        # Transform images and masks if transform is provided
        if self.transform is not None:
            transformed = self.transform(data)
            img = transformed['image']
            imu = transformed['imu_mask']

        if self.normalize_t is not None:
            img = self.normalize_t(img)
        else:
            # Default: divide by 255
            img = TF.to_tensor(img)
        imu = torch.from_numpy(imu.astype(np.bool))

        metadata_fields = ['seq', 'name']
        metadata = {field: entry[field] for field in metadata_fields}
        metadata['image_path'] = os.path.join(metadata['seq'], metadata['name'])

        features ={
            'image': img,
            'imu_mask': imu,
        }

        return features, metadata
