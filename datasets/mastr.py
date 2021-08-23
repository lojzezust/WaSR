import os
from pathlib import Path
from posixpath import splitext
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF
import yaml

def read_mask(path):
    """Reads class segmentation mask from an image file."""
    mask = np.array(Image.open(path))

    # Masks stored in RGB channels or as class ids
    if mask.ndim == 3:
        mask = mask.astype(np.float32) / 255.0
    else:
        mask = np.stack([mask==0, mask==1, mask==2], axis=-1).astype(np.float32)

    return mask

def read_image_list(path):
    """Reads image list from a file"""
    with open(path, 'r') as file:
        images = [line.strip() for line in file]
    return images

def get_image_list(image_dir):
    """Returns the list of images in the dir."""
    image_list = [os.path.splitext(img)[0] for img in os.listdir(image_dir)]
    return image_list

class MaSTr1325Dataset(torch.utils.data.Dataset):
    """MaSTr1325 dataset wrapper

    Args:
        dataset_file (str): Path to the dataset configuration file.
        transform (optional): Tranform to apply to image and masks
        normalize_t (optional): Transform that normalizes the input image
        include_original (optional): Include original (non-normalized) version of the image in the features
    """
    def __init__(self, dataset_file, transform=None, normalize_t=None, include_original=False):
        dataset_file = Path(dataset_file)
        self.dataset_dir = dataset_file.parent
        with dataset_file.open('r') as file:
            data = yaml.safe_load(file)

            # Set data directories
            self.image_dir = (self.dataset_dir / Path(data['image_dir'])).resolve()
            self.mask_dir = (self.dataset_dir / Path(data['mask_dir'])).resolve() if 'mask_dir' in data else None
            self.imu_dir = (self.dataset_dir / Path(data['imu_dir'])).resolve() if 'imu_dir' in data else None

            # Entries
            if 'image_list' in data:
                image_list = (self.dataset_dir / data['image_list']).resolve()
                self.images = read_image_list(image_list)
            else:
                self.images = get_image_list(self.image_dir)

        self.transform = transform
        self.normalize_t = normalize_t
        self.include_original = include_original

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.images[idx]
        img_path = str(self.image_dir / ('%s.jpg' % img_name))
        mask_filename = '%sm.png' % img_name

        img = np.array(Image.open(img_path))
        img_original = img

        data = {'image': img}

        if self.mask_dir is not None:
            mask_path = str(self.mask_dir / mask_filename)
            mask = read_mask(mask_path)
            data['segmentation'] = mask

        if self.imu_dir is not None:
            imu_path = str(self.imu_dir / ('%s.png' % img_name))
            imu_mask = np.array(Image.open(imu_path))
            data['imu_mask'] = imu_mask

        # Transform images and masks if transform is provided
        if self.transform is not None:
            data = self.transform(data)
            img = data['image']

        if self.normalize_t is not None:
            img = self.normalize_t(img)
        else:
            # Default: divide by 255
            img = TF.to_tensor(img)


        features = {'image': img}
        labels = {}

        if self.include_original:
            features['image_original'] = torch.from_numpy(img_original.transpose(2,0,1))

        if 'segmentation' in data:
            labels['segmentation'] = torch.from_numpy(data['segmentation'].transpose(2,0,1))

        if 'imu_mask' in data:
            features['imu_mask'] = torch.from_numpy(data['imu_mask'].astype(np.bool))

        # Add metadata to labels
        metadata = {
            'img_name': img_name,
            'mask_filename': mask_filename
        }
        labels.update(metadata)

        return features, labels
