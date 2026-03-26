# 文件名: 06_evaluate_calibration.py
# 描述: 评估校准度(ECE)和不确定性质量(UEO-max)
# python my_npc_project/06_evaluate_calibration.py

import argparse
import os
import sys
import numpy as np
import SimpleITK as sitk
import pandas as pd
from tqdm import tqdm
import nibabel as nib

# --- 1. 核心度量函数 ---

def compute_dice_coefficient(mask1, mask2):
    """
    计算两个二值掩膜的 Dice 系数。
    在这里用于计算 Error Mask 和 Uncertainty Mask 的重叠度。
    """
    mask1 = np.atleast_1d(mask1.astype(bool))
    mask2 = np.atleast_1d(mask2.astype(bool))
    
    intersection = np.count_nonzero(mask1 & mask2)
    size_i1 = np.count_nonzero(mask1)
    size_i2 = np.count_nonzero(mask2)
    
    if size_i1 + size_i2 == 0:
        return 1.0 if intersection == 0 else 0.0 # 如果两者都为空，视为完美重叠(没有错误且没有不确定性)
        
    return 2. * intersection / float(size_i1 + size_i2)

def compute_ece(probs, targets, n_bins=10, return_bins=False):
    """
    计算 Expected Calibration Error (ECE)。
    return_bins 参数，用于导出 Reliability Diagram 所需的统计量。
    """
    if len(probs) == 0:
        if return_bins:
            # 返回空统计量: count, sum_prob, sum_gt
            return 0.0, np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins)
        return 0.0
        
    # 确保范围在 [0, 1]
    probs = np.clip(probs, 0.0, 1.0)
    
    # 定义 bins 边界
    bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
    # 获取每个样本所属的 bin 索引 (0 到 n_bins-1)
    binids = np.digitize(probs, bins) - 1
    
    # 核心统计: count, sum_prob, sum_gt
    bin_total = np.bincount(binids, minlength=n_bins)
    bin_sums = np.bincount(binids, weights=probs, minlength=n_bins)
    bin_true = np.bincount(binids, weights=targets, minlength=n_bins)
    
    nonzero = bin_total != 0
    
    # 计算 ECE
    # 注意: 这里需要防止除以0，但在计算 ECE 时只关心 nonzero
    prob_true = np.zeros(n_bins)
    prob_pred = np.zeros(n_bins)
    
    prob_true[nonzero] = bin_true[nonzero] / bin_total[nonzero]
    prob_pred[nonzero] = bin_sums[nonzero] / bin_total[nonzero]
    
    total_samples = bin_total.sum()
    # ECE = sum( (bin_count / total_count) * |acc - conf| )
    if total_samples > 0:
        ece_val = np.sum((bin_total[nonzero] / total_samples) * np.abs(prob_true[nonzero] - prob_pred[nonzero]))
    else:
        ece_val = 0.0
    
    if return_bins:
        # 返回: ECE值, 每个bin的样本数, 每个bin的概率和, 每个bin的GT和
        return ece_val, bin_total, bin_sums, bin_true
    
    return ece_val

def get_uncertainty_ueo_max(error_map, uncertainty_map, roi_mask, thresholds):
    """
    计算 UEO-max (Uncertainty-Error Overlap)。
    逻辑：在 ROI 区域内，扫描不同阈值，计算 (Error) 与 (Uncertainty > t) 的 Dice，取最大值。
    """
    # 提取 ROI 内的数值以加速计算
    error_roi = error_map[roi_mask]
    unc_roi = uncertainty_map[roi_mask]
    
    if len(error_roi) == 0:
        return 0.0
        
    ueo_list = []
    
    for t in thresholds:
        # 生成二值化不确定性掩膜
        high_unc_mask = (unc_roi >= t)
        
        # 计算 Dice(Error, HighUncertainty)
        score = compute_dice_coefficient(error_roi, high_unc_mask)
        ueo_list.append(score)
        
    return max(ueo_list) if ueo_list else 0.0

# --- 2. 鲁棒读取与处理函数 ---

def robust_read_image(path):
    """读取图像，处理非正交方向矩阵问题"""
    try:
        return sitk.ReadImage(path)
    except RuntimeError as e:
        if "orthonormal" in str(e).lower():
            nii = nib.load(path)
            arr = nii.get_fdata()
            # Nibabel (H, W, D) -> SimpleITK (D, H, W) 需要转置
            if arr.ndim == 3:
                arr = arr.transpose(2, 1, 0) # z, y, x
            img = sitk.GetImageFromArray(arr)
            # 简单设置元数据，注意这里可能丢失具体空间信息，但在同一坐标系下计算 Dice/ECE 不影响
            return img
        else:
            raise e

def normalize_uncertainty(u_map, num_classes=2):
    """
    将熵归一化到 [0, 1]。
    最大熵为 log(num_classes)。
    """
    max_entropy = np.log(num_classes)
    u_norm = u_map / max_entropy
    return np.clip(u_norm, 0.0, 1.0)

# --- 3. 主评估流程 ---
def main():
    """
    python my_npc_project/06_evaluate_calibration.py
    """
    parser = argparse.ArgumentParser(description="Reliability Evaluation ")
    
    # 路径设置 (请根据你的实际情况修改默认路径)
    default_base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    
    # 数据集名称 
    # 1、调试数据：Dataset730_NPC 
    # 2、医大一： Dataset201_NPC_yidayi 
    # 3、外部验证佛山公开数据集： Dataset301_NPC_Foshan
    # 4、佛山公开数据集： Dataset302_NPC_FoshanPM
    # 5、外部验证广西： Dataset101_NPC_Guangxi
    # 6、广西： Dataset102_NPC_GuangxiPM
    dataset_name = 'Dataset201_NPC_yidayi'
    
    # default_results_path = os.path.join(default_base_path, 'nnUNet_results', dataset_name, 'EDL_UNet__EDLPlans__3d_fullres__fold0')
    # default_results_path = os.path.join(default_base_path, 'nnUNet_results', dataset_name, 'MCDropout_p0_5__EDLPlans__3d_fullres__fold0') # + mcdropout
    # default_results_path = os.path.join(default_base_path, 'nnUNet_results', dataset_name, 'DeepEnsemble_ens5__EDLPlans__3d_fullres__fold0') # + DeepEnsemble
    # default_results_path = os.path.join(default_base_path, 'nnUNet_results', dataset_name, 'TTA_inference__EDLPlans__3d_fullres__fold0') # TTA
    # default_results_path = os.path.join(default_base_path, 'nnUNet_results', dataset_name, 'DEviS__EDLPlans__3d_fullres__fold0')  # DEviS
    default_results_path = os.path.join(default_base_path, 'nnUNet_results', dataset_name, 'USIRES_gu0.1_nu0.1__EDLPlans__3d_fullres__fold0') # + USIRES
    # default_results_path = os.path.join(default_base_path, 'nnUNet_results', dataset_name, 'EDL_UNet_Posterior_temp10_0_nll1_0__EDLPlans__3d_fullres__fold0') # + hven
    # default_results_path = os.path.join(default_base_path, 'nnUNet_results', dataset_name, 'EDL_UNet_Posterior_temp10_0_nll1_0_FPCL_w0_1_t0_5__EDLPlans__3d_fullres__fold0') # + hvenfpcl

    
    parser.add_argument('--pred_folder', type=str, default=os.path.join(default_results_path, 'inference_real_final'), help="EDL 推理结果")
    # parser.add_argument('--pred_folder', type=str, default=os.path.join(default_results_path, 'inference_post'), help="hven后验预测结果路径")
    parser.add_argument('--gt_folder', type=str, default=os.path.join(default_base_path, 'nnUNet_raw', dataset_name, 'labelsTs'), help="GT 标签文件夹")
    
    # 不确定性文件名后缀
    parser.add_argument('--unc_a_suffix', type=str, default='unc_variance_aleatoric', help="偶然不确定性后缀")
    parser.add_argument('--unc_e_suffix', type=str, default='unc_variance_epistemic', help="认知不确定性后缀")
    parser.add_argument('--unc_total_suffix', type=str, default='unc_variance', help="总不确定性后缀 (MC Dropout/Deep Ensemble专用)")
    parser.add_argument('--output_csv', type=str, default=os.path.join(default_results_path, 'metrics', 'evaluation_calibration_metrics.csv'))
    parser.add_argument('--output_bins_csv', type=str, default=os.path.join(default_results_path, 'metrics', 'evaluation_calibration_bins.csv'), help="用于绘制 Reliability Diagram 的详细 Bin 统计")
    
    # 参数
    parser.add_argument('--num_classes', type=int, default=2, help="类别数，用于熵归一化")
    parser.add_argument('--bins', type=int, default=10, help="ECE 分箱数")
    
    args = parser.parse_args()
    
    # UEO 扫描阈值 
    ueo_thresholds = np.arange(0.05, 1.0, 0.05).tolist() # [0.05, 0.1, ..., 0.95]

    print(f"--- 正在评估: {dataset_name} ---")
    print(f"--- 结果目录: {args.pred_folder} ---")
    print(f"--- UEO 阈值扫描: {len(ueo_thresholds)} 个阈值 ---")

    prob_files = sorted([f for f in os.listdir(args.pred_folder) if f.endswith('_prob.nii.gz')])
    if not prob_files:
        print("错误: 未找到概率图文件 (*_prob.nii.gz)")
        return

    all_results = []
    all_bin_results = []

    for prob_file in tqdm(prob_files, desc="Processing Cases"):
        case_name = prob_file.replace('_prob.nii.gz', '')
        
        # 路径构建
        gt_file = f"{case_name.replace('_0000', '')}.nii.gz" # 根据你的GT命名规则调整
        gt_path = os.path.join(args.gt_folder, gt_file)
        prob_path = os.path.join(args.pred_folder, prob_file)
        # 动态构建路径：先尝试找 EDL 的解耦文件，如果找不到，再找 MC/Ensemble 的单文件
        unc_a_path = os.path.join(args.pred_folder, f"{case_name}_{args.unc_a_suffix}.nii.gz")
        unc_e_path = os.path.join(args.pred_folder, f"{case_name}_{args.unc_e_suffix}.nii.gz")
        unc_total_path = os.path.join(args.pred_folder, f"{case_name}_{args.unc_total_suffix}.nii.gz")
        
        # 检查文件
        mode = "unknown"
        if os.path.exists(unc_a_path) and os.path.exists(unc_e_path):
            mode = "EDL"
        elif os.path.exists(unc_total_path):
            mode = "Standard_UQ" # MC Dropout or Deep Ensemble
        else:
            print(f"警告: 病例 {case_name} 缺失不确定性文件 (既无 aleatoric/epistemic 也无 total entropy)，跳过。")
            continue
        
        if not os.path.exists(gt_path):
            print(f"警告: 病例 {case_name} 缺失 GT 文件 ({gt_path})，跳过。")
            continue
        if not os.path.exists(prob_path):
            print(f"警告: 病例 {case_name} 缺失概率图文件 ({prob_path})，跳过。")
            continue

        try:
            # 读取数据
            img_gt = robust_read_image(gt_path)
            img_prob = robust_read_image(prob_path)
            
            gt_np = sitk.GetArrayFromImage(img_gt)
            prob_np = sitk.GetArrayFromImage(img_prob)
            
            # 根据模式读取并合成 u_total_raw
            if mode == "EDL":
                img_unc_a = robust_read_image(unc_a_path)
                img_unc_e = robust_read_image(unc_e_path)
                unc_a_np = sitk.GetArrayFromImage(img_unc_a)
                unc_e_np = sitk.GetArrayFromImage(img_unc_e)
                # EDL: 总不确定性 = 偶然 + 认知
                u_total_raw = unc_a_np + unc_e_np
            else: # Standard_UQ
                img_unc_total = robust_read_image(unc_total_path)
                u_total_raw = sitk.GetArrayFromImage(img_unc_total)
            
            # 形状检查与对齐 (简单防御性检查)
            if prob_np.shape != gt_np.shape:
                # 可以在这里添加重采样逻辑，或者直接报错
                # 假设 inference 脚本已经保证了形状一致
                print(f"形状不匹配: {case_name}，跳过")
                continue

            # --- Step 1: 统一变量与 ROI 定义 ---
            
            # 1.1 基础变量
            gt = (gt_np > 0).astype(np.uint8) # 二值化 GT
            p = prob_np.astype(np.float32)    # 前景概率
            pred = (p >= 0.5).astype(np.uint8)# 预测 Mask
            
            # 1.2 定义 ROI (Union of GT and Pred)
            roi_mask = (gt == 1) | (pred == 1)
            
            if roi_mask.sum() == 0:
                print(f"警告: {case_name} 没有前景预测也没有 GT，跳过。")
                continue
                
            # 1.3 定义 Error Map (ROI 内的 FP + FN)
            error_map = (pred != gt).astype(np.uint8)

            # --- Step 2: ECE 计算 (双口径) ---
            
            # === A. ROI 区域 (严苛模式：只看病灶和预测区域) ===
            # ECE-Prob (概率校准)
            # 使用 return_bins=True 获取详细统计
            ece_p_roi, bin_total_roi, bin_sums_roi, bin_true_roi = compute_ece(
                p[roi_mask], gt[roi_mask], n_bins=args.bins, return_bins=True
            )
            
            # 保存 ROI 的 Bin 统计
            for b_idx in range(args.bins):
                all_bin_results.append({
                    'Case': case_name,
                    'ROI_Type': 'ROI',
                    'BinIdx': b_idx + 1,      # 1-based index 方便查看
                    'BinLo': b_idx / args.bins,
                    'BinHi': (b_idx + 1) / args.bins,
                    'Count': int(bin_total_roi[b_idx]),
                    'SumProb': float(bin_sums_roi[b_idx]),
                    'SumGT': float(bin_true_roi[b_idx])
                })
                
            # === B. Global 全图 (宽松模式：包含所有背景) ===
            # 注意：全图包含大量容易预测的背景(TN)，通常 ECE 会非常低
            ece_p_global, bin_total_g, bin_sums_g, bin_true_g = compute_ece(
                p.flatten(), gt.flatten(), n_bins=args.bins, return_bins=True
            )
            
            # 保存 Global 的 Bin 统计
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

            # --- Step 3: 不确定性图准备 (双口径) ---
            
            
            u_total_raw = np.nan_to_num(u_total_raw)
            u_total = normalize_uncertainty(u_total_raw, args.num_classes)
            
            epsilon = 1e-7
            p_safe = np.clip(p, epsilon, 1.0 - epsilon)
            # 注意: 只取前景项
            u_fg_raw = -(p_safe * np.log(p_safe))
            u_fg = normalize_uncertainty(u_fg_raw, args.num_classes)
            
            # --- Step 4: UEO-max 计算 ---
            
            ueo_total = get_uncertainty_ueo_max(error_map, u_total, roi_mask, ueo_thresholds)
            
            ueo_fg = get_uncertainty_ueo_max(error_map, u_fg, roi_mask, ueo_thresholds)
            
            # --- 保存结果 ---
            all_results.append({
                'Case': case_name,
                # 严苛ROI指标 (用于分析不确定性质量)
                'ECE_Prob_ROI': ece_p_roi,
                # 'ECE_Conf_ROI': ece_c_roi,
                # 宽松指标 (用于证明模型整体正常)
                'ECE_Prob_Global': ece_p_global,
                # 'ECE_Conf_Global': ece_c_global,
                # UEO 指标
                'UEO_max_Total': ueo_total, # 使用总不确定性
                'UEO_max_FG': ueo_fg        # 使用前景不确定性
            })

        except Exception as e:
            print(f"错误处理 {case_name}: {e}")
            import traceback
            traceback.print_exc()

    # --- 汇总输出 ---
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    # 1. 保存原有 Metrics
    if all_results:
        df = pd.DataFrame(all_results)
        summary = df.mean(numeric_only=True).to_frame().T
        summary['Case'] = 'Mean'
        df = pd.concat([df, summary], ignore_index=True)
        df.to_csv(args.output_csv, index=False)
        print(f"\n[Metrics] 评估完成！指标已保存至: {args.output_csv}")
        print("\n--- Summary (Mean) ---")
        print(summary.to_string(index=False))
    
    # 2. 保存新增 Bins 统计 (用于画 Reliability Diagram)
    if all_bin_results:
        df_bins = pd.DataFrame(all_bin_results)
        # df_bins = df_bins[df_bins['Count'] > 0] 
        df_bins.to_csv(args.output_bins_csv, index=False)
        print(f"\n[Bins] 绘图数据已保存至: {args.output_bins_csv}")
        print(f"数据量: {len(df_bins)} 行")

    if not all_results:
        print("未生成任何结果。")

if __name__ == "__main__":
    main()
