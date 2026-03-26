import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet
from torch.distributions.kl import kl_divergence
import torch.distributed
from typing import Dict, Union, List, Optional
import sys
import os
import math

# 导入现有的损失函数
try:
    from edl_loss import EvidentialHybridLoss, SoftDiceLoss
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from edl_loss import EvidentialHybridLoss, SoftDiceLoss


class TwoStageHVENLoss(nn.Module):
    """
    两阶段HVEN损失函数

    专门为两阶段HVEN架构设计的损失函数：
    - 阶段1：独立训练先验网络，使用NP标签监督
    - 阶段2：训练后验网络，使用GTV标签和KL约束

    关键特点：
    1. 完全解耦的两阶段训练
    2. 先验网络专注NP解剖学学习
    3. 后验网络接受先验指导进行GTV分割
    4. 单向KL约束避免相互干扰
    """

    def __init__(
        self,
        stage: int = 1,
        lambda_kl: float = 0.1,
        dice_smooth: float = 1e-5,
        prior_temperature: float = 10.0,
        dice_weight: float = 1.0,
        nll_weight: float = 0.1,
        deep_supervision_weights: Optional[List[float]] = None,
        prior_dice_weight: float = 1.0,
        prior_nll_weight: float = 0.1,
        debug_mode: bool = False,
        kl_annealing_start: int = 50,
        **kwargs
    ):
        """
        初始化两阶段HVEN损失函数

        Args:
            stage (int): 训练阶段 (1: 先验网络训练, 2: 后验网络训练)
            lambda_kl (float): KL散度损失权重
            dice_smooth (float): Dice损失平滑参数
            dice_weight (float): Dice损失权重
            nll_weight (float): NLL损失权重
            deep_supervision_weights (List[float]): 深度监督权重
            prior_dice_weight (float): 先验网络Dice损失权重
            prior_nll_weight (float): 先验网络NLL损失权重
            debug_mode (bool): 调试模式
            **kwargs: 其他参数
        """
        super(TwoStageHVENLoss, self).__init__()

        self.stage = stage
        self.lambda_kl = lambda_kl
        self.dice_weight = dice_weight
        self.prior_temperature = prior_temperature
        self.nll_weight = nll_weight
        self.prior_dice_weight = prior_dice_weight
        self.prior_nll_weight = prior_nll_weight
        self.debug_mode = debug_mode
        self.kl_annealing_start = kl_annealing_start

        # 用于保存调试信息
        self.last_dice_loss = None
        self.last_nll_loss = None
        self.last_evidence_prior = None
        self.last_evidence_update = None
        self.last_evidence_final = None

        # 创建基础损失函数
        self.dice_loss = SoftDiceLoss(smooth=dice_smooth, batch_dice=True)
        
        # 先验网络专用的软标签Dice Loss
        self.prior_soft_dice_loss = SoftDiceLoss(smooth=dice_smooth, batch_dice=True)

        

        if deep_supervision_weights is None:
            self.deep_supervision_weights = [1.0 / (2**i) for i in range(10)]
        else:
            self.deep_supervision_weights = deep_supervision_weights

        if self.debug_mode:
            print(f"[TwoStageHVENLoss] Initialized for stage {stage}")
            print(f"[TwoStageHVENLoss] KL weight: {lambda_kl}")
            print(f"[TwoStageHVENLoss] Prior temperature: {prior_temperature}")
            print(f"[TwoStageHVENLoss] Prior weights - Dice: {prior_dice_weight}, NLL: {prior_nll_weight}")

    def _compute_prior_supervision_loss_soft(
        self,
        evidence_prior: torch.Tensor,
        target_np_soft: torch.Tensor,
        nll_weight_prior: float = 1.0
    ) -> torch.Tensor:
        
        epsilon = 1e-7
        alpha_prior = evidence_prior + 1.0  # (B, 2, D, H, W)
        S_prior = torch.sum(alpha_prior, dim=1, keepdim=True)
        
        # p = alpha / S
        p_prior = alpha_prior / (S_prior + epsilon)
        
        # 分离前景概率用于 Dice
        p_prior_foreground = p_prior[:, 1:2, ...]
        dice_loss = self.prior_soft_dice_loss(p_prior_foreground, target_np_soft)
        # 构造完整的软标签 (B, 2, D, H, W)
        target_bg = 1.0 - target_np_soft
        target_fg = target_np_soft
        target_soft = torch.cat([target_bg, target_fg], dim=1)
        # 手动计算 Soft Cross Entropy: - sum(y * log(p))
        # 注意：这里 p 是投影概率
        log_p = torch.log(p_prior + epsilon)
        ce_loss = -torch.sum(target_soft * log_p, dim=1).mean()
        # 如果 Dice 之前权重是 1.0，这里 CE 权重建议设为 1.0 或 0.5
        total_loss = dice_loss + nll_weight_prior * ce_loss 
        return total_loss


    def _compute_posterior_main_loss(
        self,
        alpha_posterior: torch.Tensor,
        target_gtv: torch.Tensor,
        current_epoch: int,
        epsilon: float = 1e-7
    ) -> torch.Tensor:
        S = torch.sum(alpha_posterior, dim=1, keepdim=True)
        p_projected = alpha_posterior / (S + epsilon) 

        target_gtv_clean = target_gtv.clone()
        target_gtv_clean[target_gtv_clean < 0] = 0  

        p_foreground = p_projected[:, 1:2, ...]
        target_fg = (target_gtv_clean == 1).float()
        dice_loss_fg = self.dice_loss(p_foreground, target_fg)

        p_background = p_projected[:, 0:1, ...]
        target_bg = (target_gtv_clean == 0).float()
        dice_loss_bg = self.dice_loss(p_background, target_bg)

        dice_loss_val = (dice_loss_fg + dice_loss_bg) / 2.0
        self.last_dice_loss = dice_loss_val.item()

        # 将 GTV 标签展平用于 CE 计算
        # target_gtv shape: [B, 1, D, H, W] -> [B, D, H, W]
        target_gtv_squeeze = target_gtv_clean.squeeze(1).long()
        
        target_gtv_squeeze = torch.clamp(target_gtv_squeeze, min=0) 
        if self.debug_mode and (target_gtv < 0).any():
            print(f"[Auto-Fix] Detected -1 in targets, converted to 0 for loss calculation.")
        
        # 为了数值稳定，加 epsilon
        log_p = torch.log(p_projected + epsilon)
        
        ce_loss_val = F.nll_loss(log_p, target_gtv_squeeze)
        
        self.last_nll_loss = ce_loss_val.item() # 这里借用变量名记录 CE Loss

        total_loss = self.dice_weight * dice_loss_val + self.nll_weight * ce_loss_val

        return total_loss


    def forward_stage1(
        self,
        model_output: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
        target_np: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        阶段1：先验网络训练损失（使用软标签损失）

        Args:
            model_output (Dict): 模型输出，包含 'logits_prior'
            target_np (torch.Tensor): NP软标签，形状 (B, 1, D, H, W), 值域[0, 1]

        Returns:
            Dict[str, torch.Tensor]: 损失字典
        """
        if self.debug_mode:
            print("[TwoStageHVENLoss] === Stage 1: Prior Network Training (Soft Labels) ===")

        logits_prior = model_output['logits_prior']

        # 确保target_np是单通道格式
        if target_np.shape[1] > 1:
            target_np = target_np[:, 1:2, ...]  # 取前景通道

        if isinstance(logits_prior, list):
            # 深度监督：计算多个尺度的损失
            total_loss = 0.0
            debug_loss = 0.0

            for i, logit in enumerate(logits_prior):
                if i < len(self.deep_supervision_weights):
                    ds_weight = self.deep_supervision_weights[i]
                else:
                    ds_weight = self.deep_supervision_weights[-1]
                if ds_weight == 0:
                    continue

                # 下采样目标到当前尺度
                current_shape = logit.shape[2:]
                # 对软标签使用 trilinear 插值，保留软边界信息
                target_np_i = F.interpolate(
                    target_np.float(),
                    size=current_shape,
                    mode='trilinear',
                    align_corners=False
                )
                # 确保值域在 [0, 1]
                target_np_i = torch.clamp(target_np_i, 0.0, 1.0)

                # 将 logit 转换为 evidence
                evidence_i = F.softplus(logit)

                # 使用新的软标签损失函数
                loss_i = self._compute_prior_supervision_loss_soft(
                    evidence_i,
                    target_np_i,
                    nll_weight_prior=self.prior_nll_weight
                )
                scale_loss = ds_weight * loss_i

                total_loss += scale_loss
                if i == 0:
                    debug_loss = loss_i

            # 归一化
            norm_factor = sum(self.deep_supervision_weights[:len(logits_prior)])
            if norm_factor > 0:
                total_loss /= norm_factor

            return {
                'total_loss': total_loss,
                'prior_loss': debug_loss,
                'main_loss': torch.tensor(0.0),
                'kl_loss': torch.tensor(0.0)
            }
        else:
            # 单一输出
            current_shape = logits_prior.shape[2:]
            target_np_resized = F.interpolate(
                target_np.float(),
                size=current_shape,
                mode='trilinear',
                align_corners=False
            )
            target_np_resized = torch.clamp(target_np_resized, 0.0, 1.0)

            evidence = F.softplus(logits_prior)
            loss = self._compute_prior_supervision_loss_soft(
                evidence,
                target_np_resized,
                nll_weight_prior=self.prior_nll_weight
            )

            return {
                'total_loss': loss,
                'prior_loss': loss,
                'main_loss': torch.tensor(0.0),
                'kl_loss': torch.tensor(0.0)
            }

    def forward_stage2(
        self,
        model_output: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
        target_gtv: torch.Tensor,
        target_np: Optional[torch.Tensor] = None, 
        current_epoch: int = 0,
        total_epochs: int = 500
    ) -> Dict[str, torch.Tensor]:
        """
        阶段2：后验网络训练损失

        Args:
            model_output (Dict): 模型输出，包含 'logits_post' 和 'evidence_prior'
            target_gtv (torch.Tensor): GTV标签，形状 (B, 1, D, H, W)
            target_np (torch.Tensor): NP标签（可选，用于先验监督）
            current_epoch (int): 当前训练轮次，用于KL预热

        Returns:
            Dict[str, torch.Tensor]: 损失字典
        """
        if self.debug_mode:
            print("[TwoStageHVENLoss] === Stage 2: Posterior Network Training ===")
            print(f"[TwoStageHVENLoss] Current epoch: {current_epoch}, KL annealing start: {self.kl_annealing_start}")

        logits_posterior = model_output['logits_post']
        
        #  优先使用门控后的先验证据
        if 'gated_evidence_prior' in model_output:
            evidence_prior = model_output['gated_evidence_prior'] # 已经是 Gate * Softplus(Logits)
            if isinstance(evidence_prior, list):
                 evidence_prior_full_res = evidence_prior[0] 
            else:
                 evidence_prior_full_res = evidence_prior
        else:
            # 兼容旧逻辑
            evidence_prior = model_output['evidence_prior']

        # 确保target_gtv是单通道格式
        if target_gtv.shape[1] > 1:
            target_gtv = target_gtv[:, 0:1, ...]  # 取前景通道
        
        current_kl_weight = 0.0

        # 处理深度监督输出
        if not isinstance(logits_posterior, list):
            logits_posterior_list = [logits_posterior]
        else:
            logits_posterior_list = logits_posterior.copy()

        # logits_posterior_list.reverse()  # 从最高分辨率到最低分辨率

        # 处理先验evidence
        if isinstance(evidence_prior, list):
            evidence_prior_full_res = evidence_prior[0]  # 取最高分辨率
        else:
            evidence_prior_full_res = evidence_prior

        # 初始化损失累加器
        total_loss = 0.0
        debug_main_loss = 0.0
        debug_kl_loss = 0.0

        # 深度监督循环
        for i, logits_posterior_i in enumerate(logits_posterior_list):
            # 添加索引越界保护
            if i < len(self.deep_supervision_weights):
                ds_weight = self.deep_supervision_weights[i]
            else:
                ds_weight = self.deep_supervision_weights[-1]  # 使用最后一个权重
            if ds_weight == 0:
                continue

            # 获取当前尺度的空间维度
            current_shape = logits_posterior_i.shape[2:]

            # 下采样先验evidence到当前尺度
            evidence_prior_i = F.interpolate(
                evidence_prior_full_res,
                size=current_shape,
                mode='trilinear',
                align_corners=False
            )

            # 下采样GTV标签到当前尺度
            target_gtv_i = F.interpolate(
                target_gtv.float(),
                size=current_shape,
                mode='nearest'
            )

            # 1. 获取似然证据
            evidence_post_update = F.softplus(logits_posterior_i)

            # 2. 获取先验证据 (如果是第 0 层，直接用 Gated 的)
            if i == 0 and 'gated_evidence_prior' in model_output:
                # 使用门控先验
                # Forward 里已经乘了 Gate，这里只需要除以 T
                evidence_prior_input = evidence_prior_full_res
            else:
                evidence_prior_input = F.interpolate(
                    evidence_prior_full_res,
                    size=current_shape,
                    mode='trilinear',
                    align_corners=False)

            # 3. 应用温度 T 
            evidence_prior_scaled = evidence_prior_input / self.prior_temperature

            # 4. 应用先验退火 
            ramp_duration = 10  # 淡入期持续 10 个 epoch
            start_epoch = self.kl_annealing_start
            end_epoch = start_epoch + ramp_duration
            if current_epoch < start_epoch:
                # 阶段1: N_post 预热 (权重为0)
                prior_annealing_weight = 0.0
            elif current_epoch < end_epoch:
                # 阶段2: 先验 "淡入" (权重 0 -> 1)
                progress = (current_epoch - start_epoch) / ramp_duration
                prior_annealing_weight = progress
            else:
                # 阶段3: 完整HVEN模型 (权重为1)
                prior_annealing_weight = 1.0

            evidence_prior_annealed = evidence_prior_scaled * prior_annealing_weight

            # 5. 最终的后验 Alpha 
            #    alpha = 1 (Base Prior) + v_prior (Learned Prior) + v_post (Likelihood)
            alpha_posterior_final = 1.0 + evidence_prior_annealed + evidence_post_update

            # 保存证据值供训练脚本使用（仅保存最高分辨率的值）
            if i == 0:
                self.last_evidence_prior = evidence_prior_annealed
                self.last_evidence_update = evidence_post_update
                self.last_evidence_final = alpha_posterior_final

            # Step 4: 主损失只监督最终后验分布（唯一监督信号：GTV标签）
            main_loss_i = self._compute_posterior_main_loss(alpha_posterior_final, target_gtv_i, current_epoch=current_epoch)

            # Step 5: 移除KL损失（先验知识已通过加法注入，无需KL约束）
            kl_loss_i = torch.tensor(0.0, device=main_loss_i.device)

            # Step 6: 总损失只包含主损失
            scale_loss = ds_weight * main_loss_i
            total_loss += scale_loss

            # 记录最高分辨率的损失用于调试
            if i == 0:
                debug_main_loss = main_loss_i
                debug_kl_loss = kl_loss_i

        # 归一化总损失
        norm_factor = sum(self.deep_supervision_weights[:len(logits_posterior_list)])
        if norm_factor > 0:
            total_loss /= norm_factor

        return {
            'total_loss': total_loss,
            'main_loss': debug_main_loss,
            'prior_loss': torch.tensor(0.0),  # 阶段2不训练先验网络
            'kl_loss': debug_kl_loss,  # 固定为0
            'lambda_kl': torch.tensor(current_kl_weight)  # 固定为0
        }

    def forward(
        self,
        model_output: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
        target_gtv: Optional[torch.Tensor] = None,
        target_np: Optional[torch.Tensor] = None,
        current_epoch: int = 0,
        total_epochs: int = 500,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        前向计算损失

        Args:
            model_output (Dict): 模型输出
            target_gtv (torch.Tensor): GTV标签
            target_np (torch.Tensor): NP标签
            current_epoch (int): 当前训练轮次
            **kwargs: 其他参数

        Returns:
            Dict[str, torch.Tensor]: 损失字典
        """
        if self.stage == 1:
            if target_np is None:
                raise ValueError("Stage 1 requires target_np (NP labels)")
            return self.forward_stage1(model_output, target_np)

        elif self.stage == 2:
            if target_gtv is None:
                raise ValueError("Stage 2 requires target_gtv (GTV labels)")
            return self.forward_stage2(model_output, target_gtv, target_np, current_epoch, total_epochs)

        else:
            raise ValueError(f"Invalid stage: {self.stage}. Must be 1 or 2.")

    def set_stage(self, stage: int):
        """设置训练阶段"""
        if stage not in [1, 2]:
            raise ValueError("Stage must be 1 or 2")

        self.stage = stage
        if self.debug_mode:
            print(f"[TwoStageHVENLoss] Stage changed to {stage}")

    def get_current_stage(self) -> int:
        """获取当前训练阶段"""
        return self.stage