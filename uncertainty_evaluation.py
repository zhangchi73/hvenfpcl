import torch

class UncertaintyEvaluation:
    """
    Utilities for decomposing uncertainty from Dirichlet parameters.
    """

    @staticmethod
    def calculate_uncertainties(dirichlet_params: torch.Tensor, method: str, epsilon: float = 1e-7):
        """
        Compute uncertainty maps from a `(2, D, H, W)` Dirichlet tensor.

        Args:
            dirichlet_params: Channel 0 is background beta and channel 1 is foreground alpha.
            method: One of `variance`, `entropy`, or `vacuity`.
            epsilon: Small constant for numerical stability.

        Returns:
            A pair of `(D, H, W)` tensors.
        """
        alpha_input = dirichlet_params.unsqueeze(0)
        S = alpha_input.sum(dim=1, keepdim=True)
        p = alpha_input / S
        vacuity = 2.0 / S

        if method == 'variance':
            variance = p * (1 - p)
            S_expanded = S + 1
            EU_all_classes = variance / S_expanded
            AU_all_classes = variance - EU_all_classes

            au_foreground = AU_all_classes[:, 1, ...]
            eu_foreground = EU_all_classes[:, 1, ...]
            return au_foreground.squeeze(0), eu_foreground.squeeze(0)

        if method == 'entropy':
            entropy = -(p * torch.log(p + epsilon)).sum(dim=1)
            S_digamma = torch.digamma(S + 1)
            alpha_digamma = torch.digamma(alpha_input + 1)
            Udata_all_classes = (p * (S_digamma - alpha_digamma)).sum(dim=1)
            Udist_all_classes = entropy - Udata_all_classes
            return Udata_all_classes.squeeze(0), Udist_all_classes.squeeze(0)

        if method == 'vacuity':
            vac_map = vacuity.squeeze(0).squeeze(0)
            return vac_map, vac_map

        raise NotImplementedError(f'Uncertainty method not implemented: {method}')

    @staticmethod
    def compute_roi_statistics(pred_seg, gt_seg, uncertainty_map, expand_pixels=2):
        """
        Compute the mean uncertainty inside a ring-shaped ROI around the GTV.

        Args:
            pred_seg: Predicted segmentation map.
            gt_seg: Ground-truth segmentation map.
            uncertainty_map: Uncertainty map.
            expand_pixels: Number of dilation and erosion iterations.

        Returns:
            Mean uncertainty inside the ROI.
        """
        import numpy as np
        import torch
        from scipy.ndimage import binary_dilation, binary_erosion

        if torch.is_tensor(gt_seg):
            gt_seg = gt_seg.cpu().numpy()
        if torch.is_tensor(uncertainty_map):
            uncertainty_map = uncertainty_map.cpu().numpy()

        if gt_seg.ndim == 4 and gt_seg.shape[0] == 1:
            gt_seg = gt_seg.squeeze(0)
        if uncertainty_map.ndim == 4 and uncertainty_map.shape[0] == 1:
            uncertainty_map = uncertainty_map.squeeze(0)

        if np.sum(gt_seg) == 0:
            return 0.0

        gt_mask = (gt_seg > 0).astype(bool)
        dilated = binary_dilation(gt_mask, iterations=expand_pixels)
        eroded = binary_erosion(gt_mask, iterations=expand_pixels)
        roi_mask = dilated ^ eroded

        if not np.any(roi_mask):
            return 0.0

        roi_uncertainty = uncertainty_map[roi_mask]
        return float(np.mean(roi_uncertainty))
