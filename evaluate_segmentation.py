import argparse
import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt, binary_erosion, generate_binary_structure
import nibabel as nib

def dc(result, reference):
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))
    intersection = np.count_nonzero(result & reference)
    size_i1 = np.count_nonzero(result)
    size_i2 = np.count_nonzero(reference)
    try:
        return 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        return 0.0

def jc(result, reference):
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))
    intersection = np.count_nonzero(result & reference)
    union = np.count_nonzero(result | reference)
    try:
        return float(intersection) / float(union)
    except ZeroDivisionError:
        return 0.0

def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))
    footprint = generate_binary_structure(result.ndim, connectivity)
    
    if np.count_nonzero(result) == 0 or np.count_nonzero(reference) == 0:
        return np.array([0])

    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)
    
    if not np.any(result_border) or not np.any(reference_border):
        return np.array([0])
        
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]
    return sds

def hd95(result, reference, voxelspacing=None, connectivity=1):
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity)
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity)
    
    if hd1.size == 0 and hd2.size == 0: return 0.0
    if hd1.size == 0: return np.percentile(hd2, 95)
    if hd2.size == 0: return np.percentile(hd1, 95)
    
    return np.percentile(np.hstack((hd1, hd2)), 95)

def assd(result, reference, voxelspacing=None, connectivity=1):
    asd1 = __surface_distances(result, reference, voxelspacing, connectivity).mean()
    asd2 = __surface_distances(reference, result, voxelspacing, connectivity).mean()
    return np.mean((asd1, asd2))


def robust_read_image(path):
    """
    Read an image and fall back to Nibabel when the direction matrix is invalid.
    """
    try:
        return sitk.ReadImage(path)
    except RuntimeError as e:
        if "orthonormal" in str(e).lower():
            nii = nib.load(path)
            affine = nii.affine
            arr = nii.get_fdata()
            if arr.ndim == 3:
                arr = arr.transpose(2, 1, 0)
            img = sitk.GetImageFromArray(arr)
            spacing = np.linalg.norm(affine[:3, :3], axis=0)
            origin = affine[:3, 3]
            origin_lps = [-origin[0], -origin[1], origin[2]]
            rot_ras = affine[:3, :3] / spacing
            q, _ = np.linalg.qr(rot_ras)
            if np.linalg.det(q) < 0:
                q[:, -1] = -q[:, -1]
            m = np.diag([-1., -1., 1.])
            rot_lps = m @ q @ m
            img.SetSpacing(spacing.tolist())
            img.SetOrigin(origin_lps)
            img.SetDirection(rot_lps.flatten().tolist())
            return img
        raise e

def main():
    parser = argparse.ArgumentParser(description="Evaluate quantitative segmentation metrics.")
    default_base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

    dataset_name = 'Dataset201_NPC_yidayi'
    default_results_path = os.path.join(
        default_base_path,
        'nnUNet_results',
        dataset_name,
        'EDL_UNet_Posterior_temp10_0_nll1_0_FPCL_w0_1_t0_5__EDLPlans__3d_fullres__fold0'
    )
    parser.add_argument('--pred_folder', type=str, default=os.path.join(default_results_path, 'inference_real_final'))
    parser.add_argument('--gt_folder', type=str, default=os.path.join(default_base_path, 'nnUNet_raw', dataset_name, 'labelsTs'))
    parser.add_argument('--output_csv', type=str, default=os.path.join(default_results_path, 'metrics', 'evaluation_segmentation_metrics.csv'))
    args = parser.parse_args()

    print("--- Segmentation evaluation ---")
    
    pred_files = sorted([f for f in os.listdir(args.pred_folder) if f.endswith('_seg.nii.gz')])
    if not pred_files:
        print(f"Error: no `_seg.nii.gz` files were found in '{args.pred_folder}'.")
        return

    all_results = []
    
    for pred_file in tqdm(pred_files, desc="Evaluating cases"):
        case_name = pred_file.replace('_seg.nii.gz', '')
        gt_file_name = f"{case_name.replace('_0000', '')}.nii.gz"
        gt_path = os.path.join(args.gt_folder, gt_file_name)

        if not os.path.exists(gt_path):
            print(f"Warning: label file {gt_file_name} for case '{case_name}' was not found. Skipping.")
            continue
            
        try:
            itk_pred = robust_read_image(os.path.join(args.pred_folder, pred_file))
            itk_gt = robust_read_image(gt_path)

            pred_np_raw = sitk.GetArrayFromImage(itk_pred)
            gt_np_raw = sitk.GetArrayFromImage(itk_gt)
            if pred_np_raw.shape == gt_np_raw.shape:
                pred_np = pred_np_raw
                gt_np = gt_np_raw
                spacing = itk_gt.GetSpacing()
            else:
                print(
                    f"Warning: shape mismatch for {case_name} "
                    f"(Pred {pred_np_raw.shape} vs GT {gt_np_raw.shape}); trying physical resampling."
                )
                resampler = sitk.ResampleImageFilter()
                resampler.SetReferenceImage(itk_gt)
                resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                resampler.SetDefaultPixelValue(0)
                itk_pred_resampled = resampler.Execute(itk_pred)
                
                pred_np = sitk.GetArrayFromImage(itk_pred_resampled)
                gt_np = gt_np_raw
                spacing = itk_gt.GetSpacing()

            pred_np = (pred_np > 0).astype(np.uint8)
            gt_np = (gt_np > 0).astype(np.uint8)

            if np.sum(gt_np) == 0:
                if np.sum(pred_np) == 0:
                    dice_score, jaccard_score, hd95_score, assd_score = 1.0, 1.0, 0.0, 0.0
                else:
                    dice_score, jaccard_score = 0.0, 0.0
                    hd95_score, assd_score = 100.0, 100.0 
            else:
                dice_score = dc(pred_np, gt_np)
                jaccard_score = jc(pred_np, gt_np)
                hd95_score = hd95(pred_np, gt_np, voxelspacing=spacing)
                assd_score = assd(pred_np, gt_np, voxelspacing=spacing)

            all_results.append({
                'Case': case_name, 'Dice': dice_score, 'Jaccard (IoU)': jaccard_score,
                'HD95 (mm)': hd95_score, 'ASSD (mm)': assd_score
            })

        except Exception as e:
            print(f"Error while processing case '{case_name}': {e}")

    df = pd.DataFrame(all_results)
    output_directory = os.path.dirname(args.output_csv)
    os.makedirs(output_directory, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"\nSaved evaluation results to: {args.output_csv}")

    summary = df.describe().loc[['mean', 'std']]
    print("\n--- Evaluation summary ---")
    print(summary.round(4))

if __name__ == "__main__":
    main()
