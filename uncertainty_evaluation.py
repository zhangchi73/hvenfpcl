# --- 最终修复代码: uncertainty_evaluation.py ---
# 这个版本严格按照您提供的逻辑进行计算，并修正了其中前景/背景选择的错误。

import torch

class UncertaintyEvaluation:
    """
    基于熵的分解方法，分解的是熵entropy
    基于方差的分解方法，分解的是方差variance
    两者分解的不一样，但是都能评估方法的不确定性。
    """

    @staticmethod
    def calculate_uncertainties(dirichlet_params: torch.Tensor, method: str, epsilon: float = 1e-7):
        """
        根据您指定的数学逻辑计算不确定性。

        Args:
            dirichlet_params (torch.Tensor): 形状为 (2, D, H, W) 的张量。
                                             约定：dim=0 的第0个切片是 beta (背景)，
                                             第1个切片是 alpha (前景)。
            method (str): 'variance' 或 'entropy'。
            epsilon (float): 用于数值稳定性的微小值。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - uncertainty_map_1 (torch.Tensor): 形状为 (D, H, W)。
                - uncertainty_map_2 (torch.Tensor): 形状为 (D, H, W)。
        """
        # --- 步骤 1: 适配输入维度 ---
        # 将 (2, D, H, W) -> (1, 2, D, H, W) 以匹配您的逻辑
        alpha_input = dirichlet_params.unsqueeze(0)

        # --- 步骤 2: 精确复现您提供的计算逻辑 ---
        S = alpha_input.sum(dim=1, keepdim=True)
        p = alpha_input / S 
        
        # 计算 Vacuity (K/S)
        # alpha_input shape: (1, 2, D, H, W) -> K=2
        K = 2.0 
        vacuity = K / S  # Shape: (1, 1, D, H, W)

        if method == 'variance':
            # --- 精确复现您的方差不确定性公式 ---
            variance = p * (1 - p)  # 总方差
            S_expanded = S + 1
            EU_all_classes = variance / S_expanded  # EU = Dirichlet分布方差
            AU_all_classes = variance - EU_all_classes

            # --- 步骤 3: 修正致命的索引错误 ---
            # 选择前景类别（索引为1）的不确定性，而不是背景（索引为0）
            au_foreground = AU_all_classes[:, 1, ...]
            eu_foreground = EU_all_classes[:, 1, ...]
            
            # 返回前移除批次维度，得到 (D, H, W)
            return au_foreground.squeeze(0), eu_foreground.squeeze(0)

        elif method == 'entropy':
            # --- 精确复现您的熵不确定性公式 ---
            entropy = - (p * torch.log(p + epsilon)).sum(dim=1)
            
            S_digamma = torch.digamma(S + 1)
            alpha_digamma = torch.digamma(alpha_input + 1)
            Udata_all_classes = (p * (S_digamma - alpha_digamma)).sum(dim=1)
            
            Udist_all_classes = entropy - Udata_all_classes

            # --- 步骤 3: 对于熵，结果已经是聚合后的，直接返回 ---
            # 返回前移除批次维度，得到 (D, H, W)
            return Udata_all_classes.squeeze(0), Udist_all_classes.squeeze(0)
        
        # Vacuity 分支
        elif method == 'vacuity':
            # Vacuity 本身就是一种 Epistemic Uncertainty，没有 Aleatoric 对应项
            # 为了保持接口一致，我们返回两个相同的值，或者 (0, vacuity)
            # 这里返回 (vacuity, vacuity) 以防万一
            vac_map = vacuity.squeeze(0).squeeze(0) # (D, H, W)
            return vac_map, vac_map

        else:
            raise NotImplementedError(f'Uncertainty method not implemented: {method}')

    @staticmethod
    def compute_roi_statistics(pred_seg, gt_seg, uncertainty_map, expand_pixels=2):
        """
        计算 GTV 及其边缘区域（ROI）内的平均不确定性。

        Args:
            pred_seg: 预测的分割图（Tensor或Numpy，shape: (D,H,W) 或 (1,D,H,W)）
            gt_seg: 真实标签（Tensor或Numpy，shape: (D,H,W) 或 (1,D,H,W)）
            uncertainty_map: 不确定性图（Tensor或Numpy，shape: (D,H,W) 或 (1,D,H,W)）
            expand_pixels: 膨胀像素数，默认2

        Returns:
            float: ROI区域内的平均不确定性
        """
        import numpy as np
        import torch
        from scipy.ndimage import binary_dilation, binary_erosion

        # 转换为numpy数组
        if torch.is_tensor(gt_seg):
            gt_seg = gt_seg.cpu().numpy()
        if torch.is_tensor(uncertainty_map):
            uncertainty_map = uncertainty_map.cpu().numpy()

        # 移除批次维度（如果存在）
        if gt_seg.ndim == 4 and gt_seg.shape[0] == 1:
            gt_seg = gt_seg.squeeze(0)
        if uncertainty_map.ndim == 4 and uncertainty_map.shape[0] == 1:
            uncertainty_map = uncertainty_map.squeeze(0)

        # 如果gt_seg全为0（无肿瘤），返回0.0
        if np.sum(gt_seg) == 0:
            return 0.0

        # 创建二值掩码
        gt_mask = (gt_seg > 0).astype(bool) # 使用 bool 类型

        # 【修改】形态学操作：膨胀 - 腐蚀 = 环
        dilated = binary_dilation(gt_mask, iterations=expand_pixels)
        eroded = binary_erosion(gt_mask, iterations=expand_pixels)
        roi_mask = dilated ^ eroded  # 异或操作得到环

        # 边界检查
        if not np.any(roi_mask):
            return 0.0

        # 计算ROI区域内的平均不确定性
        roi_uncertainty = uncertainty_map[roi_mask]
        mean_uncertainty = float(np.mean(roi_uncertainty))

        return mean_uncertainty