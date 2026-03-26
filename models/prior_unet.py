import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Union, List

from .edl_unet import EDL_UNet


class Prior_UNet(nn.Module):
    """
    Independent prior network used in the two-stage HVEN pipeline.
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int = 2,
        plans: dict = None,
        configuration: str = None,
        deep_supervision: bool = True,
        debug_mode: bool = False,
        **kwargs
    ):
        """
        Initialize Prior-UNet.

        Args:
            input_channels: Number of image channels.
            num_classes: Number of segmentation classes.
            plans: nnU-Net plans dictionary.
            configuration: Configuration key in the plans dictionary.
            deep_supervision: Whether to enable deep supervision.
            debug_mode: Whether to print debug logs.
            **kwargs: Extra arguments passed to `EDL_UNet`.
        """
        super(Prior_UNet, self).__init__()

        self.debug_mode = debug_mode
        self.num_classes = num_classes

        self.unet = EDL_UNet(
            input_channels=input_channels,
            num_classes=num_classes,
            plans=plans,
            configuration=configuration,
            deep_supervision=deep_supervision,
            **kwargs
        )

        if self.debug_mode:
            print(f"[Prior_UNet] Initialized with {num_classes} classes")
            print(f"[Prior_UNet] Input channels: {input_channels}")
            print(f"[Prior_UNet] Deep supervision: {deep_supervision}")

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Run the prior UNet.

        Args:
            x: Input tensor with shape `(B, C, D, H, W)`.

        Returns:
            A single output tensor or a list of multi-scale tensors.
        """
        if self.debug_mode:
            print(f"[Prior_UNet] Forward pass started")
            print(f"[Prior_UNet] Input shape: {x.shape}")

        logits = self.unet(x)

        if self.debug_mode:
            if isinstance(logits, list):
                print(f"[Prior_UNet] Output (deep supervision): {len(logits)} scales")
                for i, logit in enumerate(logits):
                    print(f"  Scale {i}: {logit.shape}")
            else:
                print(f"[Prior_UNet] Output shape: {logits.shape}")
            print(f"[Prior_UNet] Forward pass completed")

        return logits

    def get_evidence(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Return evidence tensors for the given input.

        Args:
            x: Input tensor.

        Returns:
            Evidence tensor or list of evidence tensors.
        """
        logits = self.forward(x)

        if isinstance(logits, list):
            evidence = [F.softplus(logit) for logit in logits]
        else:
            evidence = F.softplus(logits)

        return evidence

    def freeze_all_parameters(self):
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False
        if self.debug_mode:
            print("[Prior_UNet] All parameters frozen")

    def unfreeze_all_parameters(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
        if self.debug_mode:
            print("[Prior_UNet] All parameters unfrozen")

    def get_trainable_parameters(self):
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_checkpoint(self, filepath: str, epoch: int, best_metric: float = None):
        """Save a checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'best_metric': best_metric,
            'num_parameters': self.get_trainable_parameters()
        }
        torch.save(checkpoint, filepath)
        if self.debug_mode:
            print(f"[Prior_UNet] Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str, map_location: str = None):
        """Load a checkpoint."""
        checkpoint = torch.load(filepath, map_location=map_location)
        self.load_state_dict(checkpoint['model_state_dict'])
        if self.debug_mode:
            print(f"[Prior_UNet] Checkpoint loaded from {filepath}")
            print(f"[Prior_UNet] Epoch: {checkpoint['epoch']}")
            if checkpoint['best_metric'] is not None:
                print(f"[Prior_UNet] Best metric: {checkpoint['best_metric']}")
        return checkpoint['epoch'], checkpoint.get('best_metric')
