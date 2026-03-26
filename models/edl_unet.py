import torch
import torch.nn as nn
import torch.nn.functional as F

from dynamic_network_architectures.architectures.unet import PlainConvUNet


class EDL_UNet(PlainConvUNet):
    """
    Evidential Deep Learning UNet for 3D medical image segmentation.
    """
    def __init__(
            self,
            input_channels: int,
            num_classes: int,
            plans: dict,
            configuration: str,
            deep_supervision: bool = True,
            **kwargs
        ):
        """
        Initialize the EDL UNet.

        Args:
            input_channels: Number of input channels.
            num_classes: Number of segmentation classes expected by the parent UNet.
            plans: nnU-Net plans dictionary.
            configuration: Configuration key in the plans dictionary.
            deep_supervision: Whether to enable deep supervision.
        """
        arch_params = plans["configurations"][configuration]["architecture"]["arch_kwargs"]
        
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            deep_supervision=deep_supervision,
            **arch_params
        )

        for i in range(len(self.decoder.seg_layers)):
            in_channels = self.decoder.seg_layers[i].in_channels
            
            self.decoder.seg_layers[i] = nn.Conv3d(in_channels, 2, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the network and return non-negative evidence maps.

        Args:
            x: Input tensor with shape `(B, C, D, H, W)`.

        Returns:
            Either a single evidence tensor or a list of evidence tensors.
        """
        seg_outputs = super().forward(x)

        if not isinstance(seg_outputs, list):
            return F.softplus(seg_outputs)

        evidence_outputs = []
        for output in seg_outputs:
            evidence_outputs.append(F.softplus(output))
        return evidence_outputs
