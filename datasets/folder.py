from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF

class FolderDataset(torch.utils.data.Dataset):
    """Dataset wrapper for a general directory of images (and IMU masks)."""

    def __init__(self, image_dir, imu_dir=None, normalize_t=None):
        """Creates the dataset.

        Args:
            image_dir (str): path to the image directory. Can contain arbitrary subdirectory structures.
            imu_dir (str, optional): path to the directory containing IMU masks. Should have identical structure
                                     to image_dir, except masks are stored as PNG file. Defaults to None.
            normalize_t (callable, optional): Transform used to normalize the images. Defaults to None.
        """

        self.image_dir = Path(image_dir)
        self.images = sorted([p.relative_to(image_dir) for p in Path(image_dir).glob('**/*.jpg')])
        self.imus = None
        self.imu_dir = None
        if imu_dir is not None:
            self.imu_dir = Path(imu_dir)
            self.imus = [p.with_suffix('.png') for p in self.images]

        self.normalize_t = normalize_t

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        rel_path = self.images[idx]
        img_path = self.image_dir / rel_path
        img = np.array(Image.open(str(img_path)))

        if self.normalize_t is not None:
            img = self.normalize_t(img)
        else:
            # Default: divide by 255
            img = TF.to_tensor(img)


        features = {
            'image': img
        }

        if self.imu_dir is not None:
            imu_path = self.imu_dir / self.imus[idx]
            imu = np.array(Image.open(str(imu_path)))
            imu = torch.from_numpy(imu.astype(np.bool))
            features['imu_mask'] = imu


        metadata = {
            'image_name': img_path.name,
            'image_path': str(rel_path)
        }

        return features, metadata
