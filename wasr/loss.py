import torch
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF

def water_obstacle_separation_loss(features, gt_mask):
    """Computes the water-obstacle separation loss from intermediate features.

    Args:
        features (torch.tensor): Features tensor
        gt_mask (torch.tensor): Ground truth tensor
        clipping_value (float): Clip loss at clipping_value * sigma
    """
    epsilon_watercost = 0.01
    min_samples = 5

    # Resize gt mask to match the extracted features shape (x,y)
    feature_size = (features.size(2), features.size(3))
    gt_mask = F.interpolate(gt_mask, size=feature_size, mode='area')

    # Create water and obstacles masks.
    # The masks should be of type float so we can multiply it later in order to mask the elements
    # (1 = water, 2 = sky, 0 = obstacles)
    mask_water = gt_mask[:,1].unsqueeze(1)

    mask_obstacles = gt_mask[:,0].unsqueeze(1)

    # Count number of water and obstacle pixels, clamp to at least 1 (for numerical stability)
    elements_water = mask_water.sum((0,2,3), keepdim=True).clamp(min=1.)
    elements_obstacles = mask_obstacles.sum((0,2,3), keepdim=True)

    # Zero loss if number of samples for any class is smaller than min_samples
    if elements_obstacles.squeeze() < min_samples or elements_water.squeeze() < min_samples:
        return torch.tensor(0.)

    # Only keep water and obstacle pixels. Set the rest to 0.
    water_pixels = mask_water * features
    obstacle_pixels = mask_obstacles * features

    # Mean value of water pixels per feature (batch average)
    mean_water = water_pixels.sum((0,2,3), keepdim=True) / elements_water

    # Mean water value matrices for water and obstacle pixels
    mean_water_wat = mean_water * mask_water
    mean_water_obs = mean_water * mask_obstacles

    # Variance of water pixels (per channel, batch average)
    var_water = (water_pixels - mean_water_wat).pow(2).sum((0,2,3), keepdim=True) / elements_water

    # Average quare difference of obstacle pixels and mean water values (per channel)
    difference_obs_wat = (obstacle_pixels - mean_water_obs).pow(2).sum((0,2,3), keepdim=True)

    # Compute the separation
    loss_c = elements_obstacles * var_water / (difference_obs_wat + epsilon_watercost)

    var_cost = loss_c.mean()

    return var_cost

def focal_loss(logits, labels, gamma=2.0, alpha=4.0, target_scale='labels'):
    """Focal loss of the segmentation output `logits` and ground truth `labels`."""

    epsilon = 1.e-9

    if target_scale == 'logits':
        # Resize one-hot labels to match the logits scale
        logits_size = (logits.size(2), logits.size(3))
        labels = F.interpolate(labels, size=logits_size, mode='area')
    elif target_scale == 'labels':
        # Resize network output to match the label size
        labels_size = (labels.size(2), labels.size(3))
        logits = TF.resize(logits, labels_size, interpolation=InterpolationMode.BILINEAR)
    else:
        raise ValueError('Invalid value for target_scale: %s' % target_scale)

    logits_sm = torch.softmax(logits, 1)

    # Focal loss
    fl = -labels * torch.log(logits_sm + epsilon) * (1. - logits_sm) ** gamma
    fl = fl.sum(1) # Sum focal loss along channel dimension

    # Return mean of the focal loss along spatial and batch dimensions
    return fl.mean()
