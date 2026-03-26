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

try:
    from edl_loss import EvidentialHybridLoss, SoftDiceLoss
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from edl_loss import EvidentialHybridLoss, SoftDiceLoss


class TwoStageHVENLoss(nn.Module):
    """
    Loss used by the two-stage HVEN pipeline.
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
        Initialize the two-stage HVEN loss.

        Args:
            stage: Training stage, where 1 trains the prior network and 2 trains the posterior network.
            lambda_kl: KL loss weight.
            dice_smooth: Smoothing value for Dice loss.
            dice_weight: Dice loss weight.
            nll_weight: NLL loss weight.
            deep_supervision_weights: Weights for deep supervision outputs.
            prior_dice_weight: Dice loss weight for the prior network.
            prior_nll_weight: NLL loss weight for the prior network.
            debug_mode: Whether to emit debug logs.
            **kwargs: Extra keyword arguments.
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

        self.last_dice_loss = None
        self.last_nll_loss = None
        self.last_evidence_prior = None
        self.last_evidence_update = None
        self.last_evidence_final = None

        self.dice_loss = SoftDiceLoss(smooth=dice_smooth, batch_dice=True)
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
        alpha_prior = evidence_prior + 1.0
        S_prior = torch.sum(alpha_prior, dim=1, keepdim=True)
        p_prior = alpha_prior / (S_prior + epsilon)
        p_prior_foreground = p_prior[:, 1:2, ...]
        dice_loss = self.prior_soft_dice_loss(p_prior_foreground, target_np_soft)
        target_bg = 1.0 - target_np_soft
        target_fg = target_np_soft
        target_soft = torch.cat([target_bg, target_fg], dim=1)
        log_p = torch.log(p_prior + epsilon)
        ce_loss = -torch.sum(target_soft * log_p, dim=1).mean()
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

        target_gtv_squeeze = target_gtv_clean.squeeze(1).long()
        
        target_gtv_squeeze = torch.clamp(target_gtv_squeeze, min=0) 
        if self.debug_mode and (target_gtv < 0).any():
            print(f"[Auto-Fix] Detected -1 in targets, converted to 0 for loss calculation.")
        
        log_p = torch.log(p_projected + epsilon)
        
        ce_loss_val = F.nll_loss(log_p, target_gtv_squeeze)

        self.last_nll_loss = ce_loss_val.item()

        total_loss = self.dice_weight * dice_loss_val + self.nll_weight * ce_loss_val

        return total_loss


    def forward_stage1(
        self,
        model_output: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
        target_np: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Stage 1 loss for training the prior network with soft labels.

        Args:
            model_output: Model output containing `logits_prior`.
            target_np: NP soft labels with shape `(B, 1, D, H, W)` and values in `[0, 1]`.

        Returns:
            Loss dictionary.
        """
        if self.debug_mode:
            print("[TwoStageHVENLoss] === Stage 1: Prior Network Training (Soft Labels) ===")

        logits_prior = model_output['logits_prior']

        if target_np.shape[1] > 1:
            target_np = target_np[:, 1:2, ...]

        if isinstance(logits_prior, list):
            total_loss = 0.0
            debug_loss = 0.0

            for i, logit in enumerate(logits_prior):
                if i < len(self.deep_supervision_weights):
                    ds_weight = self.deep_supervision_weights[i]
                else:
                    ds_weight = self.deep_supervision_weights[-1]
                if ds_weight == 0:
                    continue

                current_shape = logit.shape[2:]
                target_np_i = F.interpolate(
                    target_np.float(),
                    size=current_shape,
                    mode='trilinear',
                    align_corners=False
                )
                target_np_i = torch.clamp(target_np_i, 0.0, 1.0)
                evidence_i = F.softplus(logit)
                loss_i = self._compute_prior_supervision_loss_soft(
                    evidence_i,
                    target_np_i,
                    nll_weight_prior=self.prior_nll_weight
                )
                scale_loss = ds_weight * loss_i

                total_loss += scale_loss
                if i == 0:
                    debug_loss = loss_i

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
        Stage 2 loss for training the posterior network.

        Args:
            model_output: Model output containing `logits_post` and prior evidence.
            target_gtv: GTV labels with shape `(B, 1, D, H, W)`.
            target_np: Optional NP labels for prior supervision.
            current_epoch: Current epoch for schedule control.

        Returns:
            Loss dictionary.
        """
        if self.debug_mode:
            print("[TwoStageHVENLoss] === Stage 2: Posterior Network Training ===")
            print(f"[TwoStageHVENLoss] Current epoch: {current_epoch}, KL annealing start: {self.kl_annealing_start}")

        logits_posterior = model_output['logits_post']
        
        if 'gated_evidence_prior' in model_output:
            evidence_prior = model_output['gated_evidence_prior']
            if isinstance(evidence_prior, list):
                 evidence_prior_full_res = evidence_prior[0] 
            else:
                 evidence_prior_full_res = evidence_prior
        else:
            evidence_prior = model_output['evidence_prior']

        if target_gtv.shape[1] > 1:
            target_gtv = target_gtv[:, 0:1, ...]
        
        current_kl_weight = 0.0

        if not isinstance(logits_posterior, list):
            logits_posterior_list = [logits_posterior]
        else:
            logits_posterior_list = logits_posterior.copy()

        if isinstance(evidence_prior, list):
            evidence_prior_full_res = evidence_prior[0]
        else:
            evidence_prior_full_res = evidence_prior

        total_loss = 0.0
        debug_main_loss = 0.0
        debug_kl_loss = 0.0

        for i, logits_posterior_i in enumerate(logits_posterior_list):
            if i < len(self.deep_supervision_weights):
                ds_weight = self.deep_supervision_weights[i]
            else:
                ds_weight = self.deep_supervision_weights[-1]
            if ds_weight == 0:
                continue

            current_shape = logits_posterior_i.shape[2:]
            target_gtv_i = F.interpolate(
                target_gtv.float(),
                size=current_shape,
                mode='nearest'
            )

            evidence_post_update = F.softplus(logits_posterior_i)

            if i == 0 and 'gated_evidence_prior' in model_output:
                evidence_prior_input = evidence_prior_full_res
            else:
                evidence_prior_input = F.interpolate(
                    evidence_prior_full_res,
                    size=current_shape,
                    mode='trilinear',
                    align_corners=False)

            evidence_prior_scaled = evidence_prior_input / self.prior_temperature

            ramp_duration = 10
            start_epoch = self.kl_annealing_start
            end_epoch = start_epoch + ramp_duration
            if current_epoch < start_epoch:
                prior_annealing_weight = 0.0
            elif current_epoch < end_epoch:
                progress = (current_epoch - start_epoch) / ramp_duration
                prior_annealing_weight = progress
            else:
                prior_annealing_weight = 1.0

            evidence_prior_annealed = evidence_prior_scaled * prior_annealing_weight

            alpha_posterior_final = 1.0 + evidence_prior_annealed + evidence_post_update

            if i == 0:
                self.last_evidence_prior = evidence_prior_annealed
                self.last_evidence_update = evidence_post_update
                self.last_evidence_final = alpha_posterior_final

            main_loss_i = self._compute_posterior_main_loss(alpha_posterior_final, target_gtv_i, current_epoch=current_epoch)
            kl_loss_i = torch.tensor(0.0, device=main_loss_i.device)
            scale_loss = ds_weight * main_loss_i
            total_loss += scale_loss

            if i == 0:
                debug_main_loss = main_loss_i
                debug_kl_loss = kl_loss_i

        norm_factor = sum(self.deep_supervision_weights[:len(logits_posterior_list)])
        if norm_factor > 0:
            total_loss /= norm_factor

        return {
            'total_loss': total_loss,
            'main_loss': debug_main_loss,
            'prior_loss': torch.tensor(0.0),
            'kl_loss': debug_kl_loss,
            'lambda_kl': torch.tensor(current_kl_weight)
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
        Forward loss computation entry point.

        Args:
            model_output: Model outputs.
            target_gtv: GTV labels.
            target_np: NP labels.
            current_epoch: Current epoch.
            **kwargs: Extra keyword arguments.

        Returns:
            Loss dictionary.
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
        """Set the active training stage."""
        if stage not in [1, 2]:
            raise ValueError("Stage must be 1 or 2")

        self.stage = stage
        if self.debug_mode:
            print(f"[TwoStageHVENLoss] Stage changed to {stage}")

    def get_current_stage(self) -> int:
        """Get the active training stage."""
        return self.stage
