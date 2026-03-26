import argparse
import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
from tqdm import tqdm
import nibabel as nib

def compute_dice_coefficient(mask1, mask2):
    """
    Compute the Dice score between two binary masks.
    """
    mask1 = np.atleast_1d(mask1.astype(bool))
    mask2 = np.atleast_1d(mask2.astype(bool))
    
    intersection = np.count_nonzero(mask1 & mask2)
    size_i1 = np.count_nonzero(mask1)
    size_i2 = np.count_nonzero(mask2)
    
    if size_i1 + size_i2 == 0:
        return 1.0 if intersection == 0 else 0.0
        
    return 2. * intersection / float(size_i1 + size_i2)

def compute_ece(probs, targets, n_bins=10, return_bins=False):
    """
    Compute expected calibration error.
    """
    if len(probs) == 0:
        if return_bins:
            return 0.0, np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins)
        return 0.0
        
    probs = np.clip(probs, 0.0, 1.0)
    bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
    binids = np.digitize(probs, bins) - 1
    bin_total = np.bincount(binids, minlength=n_bins)
    bin_sums = np.bincount(binids, weights=probs, minlength=n_bins)
    bin_true = np.bincount(binids, weights=targets, minlength=n_bins)
    
    nonzero = bin_total != 0
    
    prob_true = np.zeros(n_bins)
    prob_pred = np.zeros(n_bins)
    
    prob_true[nonzero] = bin_true[nonzero] / bin_total[nonzero]
    prob_pred[nonzero] = bin_sums[nonzero] / bin_total[nonzero]
    
    total_samples = bin_total.sum()
    if total_samples > 0:
        ece_val = np.sum((bin_total[nonzero] / total_samples) * np.abs(prob_true[nonzero] - prob_pred[nonzero]))
    else:
        ece_val = 0.0
    
    if return_bins:
        return ece_val, bin_total, bin_sums, bin_true
    
    return ece_val

def get_uncertainty_ueo_max(error_map, uncertainty_map, roi_mask, thresholds):
    """
    Compute UEO-max by scanning multiple uncertainty thresholds inside the ROI.
    """
    error_roi = error_map[roi_mask]
    unc_roi = uncertainty_map[roi_mask]
    
    if len(error_roi) == 0:
        return 0.0
        
    ueo_list = []
    
    for t in thresholds:
        high_unc_mask = (unc_roi >= t)
        score = compute_dice_coefficient(error_roi, high_unc_mask)
        ueo_list.append(score)
        
    return max(ueo_list) if ueo_list else 0.0

def robust_read_image(path):
    """Read an image and handle invalid orientation matrices."""
    try:
        return sitk.ReadImage(path)
    except RuntimeError as e:
        if "orthonormal" in str(e).lower():
            nii = nib.load(path)
            arr = nii.get_fdata()
            if arr.ndim == 3:
                arr = arr.transpose(2, 1, 0)
            img = sitk.GetImageFromArray(arr)
            return img
        raise e

def normalize_uncertainty(u_map, num_classes=2):
    """
    Normalize entropy-like uncertainty to `[0, 1]`.
    """
    max_entropy = np.log(num_classes)
    u_norm = u_map / max_entropy
    return np.clip(u_norm, 0.0, 1.0)

def main():
    parser = argparse.ArgumentParser(description="Evaluate calibration quality and UEO.")
    
    default_base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    dataset_name = 'Dataset201_NPC_yidayi'
    default_results_path = os.path.join(
        default_base_path,
        'nnUNet_results',
        dataset_name,
        'USIRES_gu0.1_nu0.1__EDLPlans__3d_fullres__fold0'
    )

    parser.add_argument('--pred_folder', type=str, default=os.path.join(default_results_path, 'inference_real_final'), help="Prediction folder.")
    parser.add_argument('--gt_folder', type=str, default=os.path.join(default_base_path, 'nnUNet_raw', dataset_name, 'labelsTs'), help="Ground-truth folder.")
    parser.add_argument('--unc_a_suffix', type=str, default='unc_variance_aleatoric', help="Aleatoric uncertainty suffix.")
    parser.add_argument('--unc_e_suffix', type=str, default='unc_variance_epistemic', help="Epistemic uncertainty suffix.")
    parser.add_argument('--unc_total_suffix', type=str, default='unc_variance', help="Total uncertainty suffix for MC Dropout or ensembles.")
    parser.add_argument('--output_csv', type=str, default=os.path.join(default_results_path, 'metrics', 'evaluation_calibration_metrics.csv'))
    parser.add_argument('--output_bins_csv', type=str, default=os.path.join(default_results_path, 'metrics', 'evaluation_calibration_bins.csv'), help="Detailed bin statistics for reliability diagrams.")
    parser.add_argument('--num_classes', type=int, default=2, help="Number of classes for entropy normalization.")
    parser.add_argument('--bins', type=int, default=10, help="Number of ECE bins.")
    
    args = parser.parse_args()
    
    ueo_thresholds = np.arange(0.05, 1.0, 0.05).tolist()

    print(f"--- Evaluating: {dataset_name} ---")
    print(f"--- Prediction folder: {args.pred_folder} ---")
    print(f"--- UEO threshold count: {len(ueo_thresholds)} ---")

    prob_files = sorted([f for f in os.listdir(args.pred_folder) if f.endswith('_prob.nii.gz')])
    if not prob_files:
        print("Error: no probability maps matching `*_prob.nii.gz` were found.")
        return

    all_results = []
    all_bin_results = []

    for prob_file in tqdm(prob_files, desc="Processing Cases"):
        case_name = prob_file.replace('_prob.nii.gz', '')
        gt_file = f"{case_name.replace('_0000', '')}.nii.gz"
        gt_path = os.path.join(args.gt_folder, gt_file)
        prob_path = os.path.join(args.pred_folder, prob_file)
        unc_a_path = os.path.join(args.pred_folder, f"{case_name}_{args.unc_a_suffix}.nii.gz")
        unc_e_path = os.path.join(args.pred_folder, f"{case_name}_{args.unc_e_suffix}.nii.gz")
        unc_total_path = os.path.join(args.pred_folder, f"{case_name}_{args.unc_total_suffix}.nii.gz")
        mode = "unknown"
        if os.path.exists(unc_a_path) and os.path.exists(unc_e_path):
            mode = "EDL"
        elif os.path.exists(unc_total_path):
            mode = "Standard_UQ"
        else:
            print(f"Warning: uncertainty files are missing for case {case_name}. Skipping.")
            continue
        
        if not os.path.exists(gt_path):
            print(f"Warning: ground-truth file is missing for case {case_name} ({gt_path}). Skipping.")
            continue
        if not os.path.exists(prob_path):
            print(f"Warning: probability map is missing for case {case_name} ({prob_path}). Skipping.")
            continue

        try:
            img_gt = robust_read_image(gt_path)
            img_prob = robust_read_image(prob_path)
            
            gt_np = sitk.GetArrayFromImage(img_gt)
            prob_np = sitk.GetArrayFromImage(img_prob)
            
            if mode == "EDL":
                img_unc_a = robust_read_image(unc_a_path)
                img_unc_e = robust_read_image(unc_e_path)
                unc_a_np = sitk.GetArrayFromImage(img_unc_a)
                unc_e_np = sitk.GetArrayFromImage(img_unc_e)
                u_total_raw = unc_a_np + unc_e_np
            else:
                img_unc_total = robust_read_image(unc_total_path)
                u_total_raw = sitk.GetArrayFromImage(img_unc_total)
            
            if prob_np.shape != gt_np.shape:
                print(f"Shape mismatch for {case_name}; skipping.")
                continue

            gt = (gt_np > 0).astype(np.uint8)
            p = prob_np.astype(np.float32)
            pred = (p >= 0.5).astype(np.uint8)
            roi_mask = (gt == 1) | (pred == 1)
            
            if roi_mask.sum() == 0:
                print(f"Warning: case {case_name} has no foreground in prediction or GT. Skipping.")
                continue
                
            error_map = (pred != gt).astype(np.uint8)
            ece_p_roi, bin_total_roi, bin_sums_roi, bin_true_roi = compute_ece(
                p[roi_mask], gt[roi_mask], n_bins=args.bins, return_bins=True
            )
            
            for b_idx in range(args.bins):
                all_bin_results.append({
                    'Case': case_name,
                    'ROI_Type': 'ROI',
                    'BinIdx': b_idx + 1,
                    'BinLo': b_idx / args.bins,
                    'BinHi': (b_idx + 1) / args.bins,
                    'Count': int(bin_total_roi[b_idx]),
                    'SumProb': float(bin_sums_roi[b_idx]),
                    'SumGT': float(bin_true_roi[b_idx])
                })
                
            ece_p_global, bin_total_g, bin_sums_g, bin_true_g = compute_ece(
                p.flatten(), gt.flatten(), n_bins=args.bins, return_bins=True
            )
            
            for b_idx in range(args.bins):
                all_bin_results.append({
                    'Case': case_name,
                    'ROI_Type': 'Global',
                    'BinIdx': b_idx + 1,
                    'BinLo': b_idx / args.bins,
                    'BinHi': (b_idx + 1) / args.bins,
                    'Count': int(bin_total_g[b_idx]),
                    'SumProb': float(bin_sums_g[b_idx]),
                    'SumGT': float(bin_true_g[b_idx])
                })

            u_total_raw = np.nan_to_num(u_total_raw)
            u_total = normalize_uncertainty(u_total_raw, args.num_classes)
            
            epsilon = 1e-7
            p_safe = np.clip(p, epsilon, 1.0 - epsilon)
            u_fg_raw = -(p_safe * np.log(p_safe))
            u_fg = normalize_uncertainty(u_fg_raw, args.num_classes)
            
            ueo_total = get_uncertainty_ueo_max(error_map, u_total, roi_mask, ueo_thresholds)
            
            ueo_fg = get_uncertainty_ueo_max(error_map, u_fg, roi_mask, ueo_thresholds)
            
            all_results.append({
                'Case': case_name,
                'ECE_Prob_ROI': ece_p_roi,
                'ECE_Prob_Global': ece_p_global,
                'UEO_max_Total': ueo_total,
                'UEO_max_FG': ueo_fg
            })

        except Exception as e:
            print(f"Error while processing {case_name}: {e}")
            import traceback
            traceback.print_exc()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    if all_results:
        df = pd.DataFrame(all_results)
        summary = df.mean(numeric_only=True).to_frame().T
        summary['Case'] = 'Mean'
        df = pd.concat([df, summary], ignore_index=True)
        df.to_csv(args.output_csv, index=False)
        print(f"\n[Metrics] Saved calibration metrics to: {args.output_csv}")
        print("\n--- Summary (Mean) ---")
        print(summary.to_string(index=False))
    
    if all_bin_results:
        df_bins = pd.DataFrame(all_bin_results)
        df_bins.to_csv(args.output_bins_csv, index=False)
        print(f"\n[Bins] Saved reliability-diagram statistics to: {args.output_bins_csv}")
        print(f"Rows: {len(df_bins)}")

    if not all_results:
        print("No results were generated.")

if __name__ == "__main__":
    main()
