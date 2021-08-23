from PIL import Image

import torch
from torch.optim.lr_scheduler import LambdaLR
import torchvision.transforms.functional as TF
import pytorch_lightning as pl

from .loss import focal_loss, water_obstacle_separation_loss
from .metrics import PixelAccuracy, ClassIoU

NUM_EPOCHS = 50
LEARNING_RATE = 1e-6
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-6
LR_DECAY_POW = 0.9
FOCAL_LOSS_SCALE = 'labels'
SL_LAMBDA = 0.01

class LitModel(pl.LightningModule):
    """ Pytorch Lightning wrapper for a model, ready for distributed training. """

    @staticmethod
    def add_argparse_args(parser):
        """Adds model specific parameters to parser."""

        parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE,
                            help="Base learning rate for training with polynomial decay.")
        parser.add_argument("--momentum", type=float, default=MOMENTUM,
                            help="Momentum component of the optimiser.")
        parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                            help="Number of training epochs.")
        parser.add_argument("--lr_decay_pow", type=float, default=LR_DECAY_POW,
                            help="Decay parameter to compute the learning rate decay.")
        parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY,
                            help="Regularisation parameter for L2-loss.")
        parser.add_argument("--focal_loss_scale", type=str, default=FOCAL_LOSS_SCALE, choices=['logits', 'labels'],
                            help="Which scale to use for focal loss computation (logits or labels).")
        parser.add_argument("--no_separation_loss", action='store_true', help="Disable separation loss.")
        parser.add_argument("--separation_loss_lambda", default=SL_LAMBDA, type=float,
                            help="The separation loss lambda (weight).")

        return parser

    def __init__(self, model, num_classes, args):
        super().__init__()

        self.model = model
        self.num_classes = num_classes

        self.epochs = args.epochs
        self.learning_rate = args.learning_rate
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        self.lr_decay_pow = args.lr_decay_pow
        self.focal_loss_scale = args.focal_loss_scale
        self.separation_loss = not args.no_separation_loss
        self.separation_loss_lambda = args.separation_loss_lambda

        # Metrics
        self.val_accuracy = PixelAccuracy(num_classes)
        self.val_iou_0 = ClassIoU(0, num_classes)
        self.val_iou_1 = ClassIoU(1, num_classes)
        self.val_iou_2 = ClassIoU(2, num_classes)

    def forward(self, x):
        output = self.model(x)
        return output['out']

    def training_step(self, batch, batch_idx):
        features, labels = batch

        out = self.model(features)

        fl = focal_loss(out['out'], labels['segmentation'], target_scale=self.focal_loss_scale)

        if self.separation_loss:
            separation_loss = water_obstacle_separation_loss(out['aux'], labels['segmentation'])
        else:
            separation_loss = torch.tensor(0.0)

        separation_loss = self.separation_loss_lambda * separation_loss
        loss = fl + separation_loss

        # log losses
        self.log('train/loss', loss.item())
        self.log('train/focal_loss', fl.item())
        self.log('train/separation_loss', separation_loss.item())

        return loss

    # def training_epoch_end(self, outputs):
    #     # Bugfix
    #     pass

    def validation_step(self, batch, batch_idx):
        features, labels = batch

        out = self.model(features)

        loss = focal_loss(out['out'], labels['segmentation'], target_scale=self.focal_loss_scale)

        # Log loss
        self.log('val/loss', loss.item())

        # Metrics
        labels_size = (labels['segmentation'].size(2), labels['segmentation'].size(3))
        logits = TF.resize(out['out'], labels_size, interpolation=Image.BILINEAR)
        preds = logits.argmax(1)

        # Create hard labels from soft
        labels_hard = labels['segmentation'].argmax(1)
        ignore_mask = labels['segmentation'].sum(1) < 0.9
        labels_hard = labels_hard * ~ignore_mask + 4 * ignore_mask

        self.val_accuracy(preds, labels_hard)
        self.val_iou_0(preds, labels_hard)
        self.val_iou_1(preds, labels_hard)
        self.val_iou_2(preds, labels_hard)

        self.log('val/accuracy', self.val_accuracy)
        self.log('val/iou/obstacle', self.val_iou_0)
        self.log('val/iou/water', self.val_iou_1)
        self.log('val/iou/sky', self.val_iou_2)

        return {'loss': loss, 'preds': preds}

    def configure_optimizers(self):
        # Separate parameters for different LRs
        encoder_parameters = []
        decoder_w_parameters = []
        decoder_b_parameters = []
        for name, parameter in self.model.named_parameters():
            if name.startswith('backbone'):
                encoder_parameters.append(parameter)
            elif 'weight' in name:
                decoder_w_parameters.append(parameter)
            else:
                decoder_b_parameters.append(parameter)

        optimizer = torch.optim.RMSprop([
            {'params': encoder_parameters, 'lr': self.learning_rate},
            {'params': decoder_w_parameters, 'lr': self.learning_rate * 10},
            {'params': decoder_b_parameters, 'lr': self.learning_rate * 20},
        ], momentum=self.momentum, alpha=0.9, weight_decay=self.weight_decay)

        # Decaying LR function
        lr_fn = lambda epoch: (1 - epoch/self.epochs) ** self.lr_decay_pow

        # Decaying learning rate (updated each epoch)
        scheduler = LambdaLR(optimizer, lr_fn)

        return [optimizer], [scheduler]

    def on_save_checkpoint(self, checkpoint):
        # Export the model weights
        checkpoint['model'] = self.model.state_dict()
