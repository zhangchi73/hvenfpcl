import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Union, List

from .edl_unet import EDL_UNet


class Prior_UNet(nn.Module):
    """
    完全独立的先验网络 (Prior-UNet)

    这是两阶段HVEN架构的核心组件之一：
    1. 完全独立于后验网络，拥有自己的编码器-解码器结构
    2. 专门学习鼻咽部(NP)的解剖学先验
    3. 输出2通道evidence（前景/背景的Beta分布参数）
    4. 在第一阶段独立训练，第二阶段被冻结用于指导后验网络

    架构特点：
    - 基于EDL_UNet但完全独立运行
    - 输出evidence而非直接的概率
    - 支持深度监督
    - 使用NP标签进行监督训练
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int = 2,  # 固定为2类：前景/背景
        plans: dict = None,
        configuration: str = None,
        deep_supervision: bool = True,
        debug_mode: bool = False,
        **kwargs
    ):
        """
        初始化Prior-UNet

        Args:
            input_channels (int): 输入图像通道数
            num_classes (int): 分割类别数，固定为2
            plans (dict): nnU-Net规划参数
            configuration (str): 配置名称
            deep_supervision (bool): 是否启用深度监督
            debug_mode (bool): 调试模式
            **kwargs: 其他传递给EDL_UNet的参数
        """
        super(Prior_UNet, self).__init__()

        self.debug_mode = debug_mode
        self.num_classes = num_classes

        # 创建完全独立的UNet架构
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
        前向传播

        Args:
            x (torch.Tensor): 输入3D图像，形状 (B, C, D, H, W)

        Returns:
            Union[torch.Tensor, List[torch.Tensor]]:
                - 如果deep_supervision=False: 返回单个logits张量 (B, 2, D, H, W)
                - 如果deep_supervision=True: 返回多尺度logits列表
        """
        if self.debug_mode:
            print(f"[Prior_UNet] Forward pass started")
            print(f"[Prior_UNet] Input shape: {x.shape}")

        # 通过独立的UNet进行前向传播
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
        获取evidence输出（logits经过softplus）

        Args:
            x (torch.Tensor): 输入图像

        Returns:
            Union[torch.Tensor, List[torch.Tensor]]: evidence张量
        """
        logits = self.forward(x)

        if isinstance(logits, list):
            # 深度监督：对每个尺度都计算evidence
            evidence = [F.softplus(logit) for logit in logits]
        else:
            # 单一输出
            evidence = F.softplus(logits)

        return evidence

    def freeze_all_parameters(self):
        """冻结所有参数（用于第二阶段）"""
        for param in self.parameters():
            param.requires_grad = False
        if self.debug_mode:
            print("[Prior_UNet] All parameters frozen")

    def unfreeze_all_parameters(self):
        """解冻所有参数"""
        for param in self.parameters():
            param.requires_grad = True
        if self.debug_mode:
            print("[Prior_UNet] All parameters unfrozen")

    def get_trainable_parameters(self):
        """获取可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_checkpoint(self, filepath: str, epoch: int, best_metric: float = None):
        """保存模型checkpoint"""
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
        """加载模型checkpoint"""
        checkpoint = torch.load(filepath, map_location=map_location)
        self.load_state_dict(checkpoint['model_state_dict'])
        if self.debug_mode:
            print(f"[Prior_UNet] Checkpoint loaded from {filepath}")
            print(f"[Prior_UNet] Epoch: {checkpoint['epoch']}")
            if checkpoint['best_metric'] is not None:
                print(f"[Prior_UNet] Best metric: {checkpoint['best_metric']}")
        return checkpoint['epoch'], checkpoint.get('best_metric')