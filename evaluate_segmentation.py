# 文件名: 05_evaluate_segmentation.py
# python my_npc_project/05_evaluate_segmentation.py


import argparse
import os
import sys
import numpy as np
import SimpleITK as sitk
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt, binary_erosion, generate_binary_structure
import nibabel as nib

# --- 核心评估函数 (保持不变) ---
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


# ---  鲁棒读取函数 ---
def robust_read_image(path):
    """
    尝试使用 SimpleITK 读取。如果因非正交方向矩阵失败，
    则使用 Nibabel 读取并强制正交化。
    """
    try:
        return sitk.ReadImage(path)
    except RuntimeError as e:
        if "orthonormal" in str(e).lower():
            # 使用 Nibabel 作为回退方案
            nii = nib.load(path)
            affine = nii.affine
            
            # 1. 提取数据并转置 (Nibabel [x,y,z] -> SimpleITK [z,y,x])
            arr = nii.get_fdata()
            if arr.ndim == 3:
                arr = arr.transpose(2, 1, 0)
            
            # 创建图像
            img = sitk.GetImageFromArray(arr)
            
            # 2. 提取几何信息并转换坐标系 (RAS -> LPS)
            # Spacing (列向量模长)
            spacing = np.linalg.norm(affine[:3, :3], axis=0)
            
            # Origin (RAS -> LPS: x,y 取反)
            origin = affine[:3, 3]
            origin_lps = [-origin[0], -origin[1], origin[2]]
            
            # Direction (RAS -> LPS + 正交化)
            # 提取旋转部分
            rot_ras = affine[:3, :3] / spacing
            
            # 使用 QR 分解强制正交化 (Q 是正交矩阵)
            q, r = np.linalg.qr(rot_ras)
            # 确保行列式为正（保持手性）
            if np.linalg.det(q) < 0:
                q[:, -1] = -q[:, -1]
            
            # 转换坐标系矩阵 M = diag(-1, -1, 1)
            m = np.diag([-1., -1., 1.])
            rot_lps = m @ q @ m
            
            # 设置元数据
            img.SetSpacing(spacing.tolist())
            img.SetOrigin(origin_lps)
            img.SetDirection(rot_lps.flatten().tolist())
            
            return img
        else:
            raise e

# --- 主评估流程 ---
def main():
    '''
    python my_npc_project/05_evaluate_segmentation.py
    '''
    parser = argparse.ArgumentParser(description="评估分割性能的定量指标")
    default_base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

    # 数据集名称 
    # 1、调试数据：Dataset730_NPC 
    # 2、医大一： Dataset201_NPC_yidayi 
    # 3、外部验证佛山公开数据集： Dataset301_NPC_Foshan
    # 4、佛山公开数据集： Dataset302_NPC_FoshanPM
    # 5、外部验证广西： Dataset101_NPC_Guangxi
    # 6、广西： Dataset102_NPC_GuangxiPM
    dataset_name = 'Dataset201_NPC_yidayi'

    # ----------------------------------------------------
    # default_results_path = os.path.join(default_base_path, 'nnUNet_results', dataset_name, 'EDL_UNet__EDLPlans__3d_fullres__fold0')
    # default_results_path = os.path.join(default_base_path, 'nnUNet_results', dataset_name, 'EDL_UNet_ratio0_08__EDLPlans__3d_fullres__fold0') # 不同比例
    # default_results_path = os.path.join(default_base_path, 'nnUNet_results', dataset_name, 'EDL_UNet_Standard__EDLPlans__3d_fullres__fold0') # + 对比模块
    # default_results_path = os.path.join(default_base_path, 'nnUNet_results', dataset_name, 'EDL_UNet_EDL_FPCL_fw0_1_t0_5__EDLPlans__3d_fullres__fold0') # EDL + FPCL
    # default_results_path = os.path.join(default_base_path, 'nnUNet_results', dataset_name, 'ablation', 'edl_fpcl', 'Exp3', 'fold0') # EDL + FPCL - EU
    # default_results_path = os.path.join(default_base_path, 'nnUNet_results', dataset_name, 'EDL_UNet_Posterior_temp10_0_nll1_0__EDLPlans__3d_fullres__fold0') # + hven
    # default_results_path = os.path.join(default_base_path, 'nnUNet_results', dataset_name, 'EDL_UNet_Posterior_temp10_0_nll1_0_ratio0_6__EDLPlans__3d_fullres__fold0') # + hven不同比例
    default_results_path = os.path.join(default_base_path, 'nnUNet_results', dataset_name, 'EDL_UNet_Posterior_temp10_0_nll1_0_FPCL_w0_1_t0_5__EDLPlans__3d_fullres__fold0') # + hvenfpcl
    # default_results_path = os.path.join(default_base_path, 'nnUNet_results', dataset_name, 'MCDropout_p0_5__EDLPlans__3d_fullres__fold0') # + mcdropout
    # default_results_path = os.path.join(default_base_path, 'nnUNet_results', dataset_name, 'DEviS__EDLPlans__3d_fullres__fold0') # + DEviS
    # default_results_path = os.path.join(default_base_path, 'nnUNet_results', dataset_name, 'USIRES_gu0.1_nu0.1__EDLPlans__3d_fullres__fold0') # + USIRES
    # default_results_path = os.path.join(default_base_path, 'nnUNet_results', dataset_name, 'TTA_inference__EDLPlans__3d_fullres__fold0') # TTA
    # default_results_path = os.path.join(default_base_path, 'nnUNet_results', dataset_name, 'DeepEnsemble_ens5__EDLPlans__3d_fullres__fold0') # + DeepEnsemble
    # default_results_path = os.path.join(default_base_path, 'nnUNet_results', dataset_name, 'PlainUNet__EDLPlans__3d_fullres__fold0') # + nnUNet
    # default_results_path = os.path.join(default_base_path, 'nnUNet_results', dataset_name, 'ablation', 'edl_fpcl', 'Exp3', 'fold0') # + ablation
    
    parser.add_argument('--pred_folder', type=str, default=os.path.join(default_results_path, 'inference_real_final'))
    # parser.add_argument('--pred_folder', type=str, default=os.path.join(default_results_path, 'inference_post'))    # hven
    parser.add_argument('--gt_folder', type=str, default=os.path.join(default_base_path, 'nnUNet_raw', dataset_name, 'labelsTs'))
    # ----------------------------------------------------

    parser.add_argument('--output_csv', type=str, default=os.path.join(default_results_path, 'metrics','evaluation_segmentation_metrics.csv')) 
    args = parser.parse_args()

    print("--- 评估脚本: 分割性能评估 ---")
    
    pred_files = sorted([f for f in os.listdir(args.pred_folder) if f.endswith('_seg.nii.gz')])
    if not pred_files:
        print(f"错误: 在 '{args.pred_folder}' 中未找到 _seg.nii.gz 文件。")
        return

    all_results = []
    
    for pred_file in tqdm(pred_files, desc="正在评估病例"):
        case_name = pred_file.replace('_seg.nii.gz', '')
        gt_file_name = f"{case_name.replace('_0000', '')}.nii.gz"
        gt_path = os.path.join(args.gt_folder, gt_file_name)

        if not os.path.exists(gt_path):
            print(f"警告: 找不到病例 '{case_name}' 的标签 {gt_file_name}。跳过。")
            continue
            
        try:
            # 使用之前定义的 robust_read_image 读取
            itk_pred = robust_read_image(os.path.join(args.pred_folder, pred_file))
            itk_gt = robust_read_image(gt_path)

            # 1. 获取原始 Numpy 数组
            pred_np_raw = sitk.GetArrayFromImage(itk_pred)
            gt_np_raw = sitk.GetArrayFromImage(itk_gt)

            # [DEBUG] 打印唯一值，检查预测是否全黑
            # unique_pred = np.unique(pred_np_raw)
            # print(f"Case {case_name}: Pred Unique Values: {unique_pred}")

            # 2. 形状检查与强制对齐
            # 如果形状完全一致，直接忽略物理头信息，进行数组比较（最稳妥的方案）
            if pred_np_raw.shape == gt_np_raw.shape:
                pred_np = pred_np_raw
                gt_np = gt_np_raw
                spacing = itk_gt.GetSpacing() # 计算HD95仍需Spacing，借用GT的
            else:
                # 只有形状不一致时，才被迫使用重采样
                print(f"警告: 形状不匹配 {case_name} (Pred {pred_np_raw.shape} vs GT {gt_np_raw.shape})，尝试物理重采样...")
                resampler = sitk.ResampleImageFilter()
                resampler.SetReferenceImage(itk_gt)
                resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                resampler.SetDefaultPixelValue(0)
                itk_pred_resampled = resampler.Execute(itk_pred)
                
                pred_np = sitk.GetArrayFromImage(itk_pred_resampled)
                gt_np = gt_np_raw
                spacing = itk_gt.GetSpacing()

            # 3. 二值化 (确保只有 0 和 1)
            pred_np = (pred_np > 0).astype(np.uint8)
            gt_np = (gt_np > 0).astype(np.uint8)

            # 检查是否有前景像素，避免全黑图像的计算问题
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
            
            # 打印单个病例结果，方便实时监控
            # print(f"Case {case_name}: Dice {dice_score:.4f}")

            all_results.append({
                'Case': case_name, 'Dice': dice_score, 'Jaccard (IoU)': jaccard_score,
                'HD95 (mm)': hd95_score, 'ASSD (mm)': assd_score
            })

        except Exception as e:
            print(f"处理病例 '{case_name}' 时发生错误: {e}")

    df = pd.DataFrame(all_results)
    output_directory = os.path.dirname(args.output_csv)
    os.makedirs(output_directory, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"\n最终的、经过校准的评估结果已保存到: {args.output_csv}")

    summary = df.describe().loc[['mean', 'std']]
    print("\n--- 最终评估结果摘要 ---")
    print(summary.round(4))

if __name__ == "__main__":
    main()