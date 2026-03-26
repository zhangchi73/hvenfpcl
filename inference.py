# python my_npc_project/04_inference.py

import argparse
import os
import sys
import json
import numpy as np
import torch
from tqdm import tqdm
from collections import OrderedDict
import SimpleITK as sitk
import importlib
from queue import Queue
from threading import Thread
from typing import Union, List
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

# --- nnU-Net v2 核心组件 ---
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.inference.sliding_window_prediction import compute_gaussian

# --- 项目自定义组件 ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.edl_unet import EDL_UNet
from uncertainty_evaluation import UncertaintyEvaluation
from utils.augment import apply_perturbation

def string_to_class(path_to_class: str):
    """辅助函数，用于将字符串转换为Python类。"""
    try:
        module_path, class_name = path_to_class.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except Exception as e:
        print(f"无法从字符串 '{path_to_class}' 解析类")
        raise e

class EDLUNetPredictor(nnUNetPredictor):
    """
    为EDL-UNet定制的最终版预测器。
    它使用 nnU-Net 的预处理和滑窗推理引擎，集成了自定义的后处理和保存逻辑，
    并通过重写核心方法解决了所有数值稳定性问题。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preprocessor = None
        self.configuration_name = None

    def initialize_from_paths(self, model_path: str, plans_path: str, dataset_json_path: str, configuration_name: str = '3d_fullres'):
        print("--- 1a. 手动加载配置和Plans ---")
        if not os.path.exists(plans_path): raise FileNotFoundError(f"Plans file not found at {plans_path}")
        if not os.path.exists(dataset_json_path): raise FileNotFoundError(f"Dataset JSON file not found at {dataset_json_path}")
        
        self.configuration_name = configuration_name
        
        self.plans_manager = PlansManager(plans_path)
        with open(dataset_json_path, 'r') as f: self.dataset_json = json.load(f)
        
        self.configuration_manager = self.plans_manager.get_configuration(self.configuration_name)
        self.label_manager = self.plans_manager.get_label_manager(self.dataset_json)
        
        preprocessor_class = self.configuration_manager.preprocessor_class
        self.preprocessor = preprocessor_class(verbose=self.verbose_preprocessing)
        
        print("--- 1b. 加载自定义EDL-UNet模型 ---")
        # [修复开始] 直接从 plans 字典中读取配置，避免 AttributeError
        raw_config = self.plans_manager.plans['configurations'][self.configuration_name]
        
        # 1. 获取架构参数字典
        arch_kwargs = raw_config['architecture']['arch_kwargs']
        
        # 2. 获取需要动态导入的参数键名
        # 检查是否存在 '_kw_requires_import' 键 (这是 standard nnU-Net v2 plans 的标准键名)
        keys_to_import = raw_config['architecture'].get('_kw_requires_import', [])

        for key in keys_to_import:
             if arch_kwargs.get(key) and isinstance(arch_kwargs[key], str): 
                 arch_kwargs[key] = string_to_class(arch_kwargs[key])
        
        self.plans_manager.plans['configurations'][self.configuration_name]['architecture']['arch_kwargs'] = arch_kwargs

        self.network = EDL_UNet(
            input_channels=len(self.dataset_json['channel_names']),
            num_classes=len(self.dataset_json['labels']),
            plans=self.plans_manager.plans,
            configuration=self.configuration_name,
            deep_supervision=True
        )
        
        state_dict = torch.load(model_path, map_location='cpu')
        new_state_dict = OrderedDict((k[7:] if k.startswith('module.') else k, v) for k, v in state_dict.items())
        self.network.load_state_dict(new_state_dict)
        self.network.to(self.device).eval()
        self.list_of_parameters = [self.network.state_dict()]
        print("模型和配置加载成功！")

    def _export_tensor_to_nifti(self, tensor_data: torch.Tensor, properties: dict, output_path: str, is_segmentation: bool):
        """
        一个完全正确的导出函数，能将任何给定的图（分割、概率等）
        正确地逆转所有预处理步骤，并保存到原始图像空间。
        """
        numpy_array = tensor_data.numpy()
        
        # 确保输入符合预期
        assert numpy_array.ndim == 4, f"导出函数 _export_tensor_to_nifti 期望一个4D数组 (C,D,H,W)，但收到了 {numpy_array.ndim}D 数组"
        
        plans = self.plans_manager.plans['configurations'][self.configuration_name]

        # 步骤 1: 反向重采样 (Revert Resampling)
        if is_segmentation:
            resampling_fn = self.configuration_manager.resampling_fn_seg
            resampling_kwargs = plans['resampling_fn_seg_kwargs']
        else:
            resampling_fn = self.configuration_manager.resampling_fn_probabilities
            resampling_kwargs = plans['resampling_fn_probabilities_kwargs']

        resampled_array = resampling_fn(
            data=numpy_array,
            new_shape=properties['shape_after_cropping_and_before_resampling'],
            current_spacing=self.configuration_manager.spacing,
            new_spacing=properties['spacing'],
            **resampling_kwargs
        )
        
        # 步骤 2: 反向裁剪 (Revert Cropping)
        shape_before_cropping = properties['shape_before_cropping']
        
        # resampled_array 仍然是 4D (C, D', H', W')
        # 为反向裁剪创建最终大小的容器
        num_channels = resampled_array.shape[0]
        final_shape = (num_channels, *shape_before_cropping)
            
        reverted_cropping_np = np.zeros(final_shape, dtype=resampled_array.dtype)
        # 将重采样后的数据插入
        reverted_cropping_np = insert_crop_into_image(reverted_cropping_np, resampled_array, properties['bbox_used_for_cropping'])
        
        # # --- 如果是分割图，此时我们才可以将它从 (1, Z, Y, X) 转换为 (Z, Y, X) ---
        # if is_segmentation:
        #     # SimpleITK.GetImageFromArray 会正确处理从 (Z,Y,X) numpy 数组到 ITK 图像的转换
        #     # 我们必须在转置和传递给sitk之前移除通道维度
        #     reverted_cropping_np = reverted_cropping_np.squeeze(0)
        if reverted_cropping_np.shape[0] == 1:
            reverted_cropping_np = reverted_cropping_np.squeeze(0)

        # 步骤 3: 反向转置 (Revert Transposition)
        transpose_backward = self.plans_manager.transpose_backward
        
        if reverted_cropping_np.ndim == 4: # 针对概率图等多通道图
            final_np_array = reverted_cropping_np.transpose([0] + [i + 1 for i in transpose_backward])
        else: # 针对已经被squeeze的分割图
            final_np_array = reverted_cropping_np.transpose(transpose_backward)

        # 步骤 4: 保存为 NifTI
        if is_segmentation:
            final_np_array = final_np_array.astype(np.uint8)
        else:
            final_np_array = final_np_array.astype(np.float32)

        itk_image = sitk.GetImageFromArray(final_np_array)
        # 1. 尝试使用 sitk_stuff (旧版或 SimpleITKIO)
        if 'sitk_stuff' in properties:
            sitk_stuff = properties['sitk_stuff']
            itk_image.SetSpacing(sitk_stuff['spacing'])
            itk_image.SetOrigin(sitk_stuff['origin'])
            itk_image.SetDirection(sitk_stuff['direction'])
            
        # 2. 尝试使用 nibabel_stuff (NibabelIO) - 需要转换坐标系
        elif 'nibabel_stuff' in properties:
            stuff = properties['nibabel_stuff']
            
            # 尝试获取仿射矩阵
            if isinstance(stuff, dict):
                affine = stuff.get('original_affine')
                if affine is None:
                    affine = stuff.get('affine')
            else:
                affine = stuff # 假设直接是矩阵
                
            if affine is None:
                print(f"Warning: Found nibabel_stuff but could not extract affine. Using identity.")
                affine = np.eye(4)

            # 从 RAS (Nibabel) 转换到 LPS (SimpleITK)
            # 1. Spacing
            spacing = np.linalg.norm(affine[:3, :3], axis=0)
            
            # 2. Origin: RAS -> LPS (x -> -x, y -> -y)
            origin_ras = affine[:3, 3]
            origin_lps = [-origin_ras[0], -origin_ras[1], origin_ras[2]]
            
            # 3. Direction: RAS -> LPS
            rot_mat = affine[:3, :3] / spacing
            # 转换矩阵 M = diag(-1, -1, 1)
            conv = np.diag([-1., -1., 1.])
            direction_lps = conv @ rot_mat @ conv
            
            itk_image.SetSpacing(spacing.tolist())
            itk_image.SetOrigin(origin_lps)
            itk_image.SetDirection(direction_lps.flatten().tolist())

        # 3. 尝试使用 nnU-Net v2 标准键 (itk_*)
        else:
            # 优先查找带前缀的，其次查找不带前缀的，最后使用默认值
            spacing = properties.get('itk_spacing', properties.get('spacing', [1.0, 1.0, 1.0]))
            origin = properties.get('itk_origin', properties.get('origin', [0.0, 0.0, 0.0]))
            direction = properties.get('itk_direction', properties.get('direction', [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]))
            
            itk_image.SetSpacing(spacing)
            itk_image.SetOrigin(origin)
            itk_image.SetDirection(direction)
        # ========================= 【核心修复结束】 =========================
        sitk.WriteImage(itk_image, output_path, True)

    def predict_case(self, input_path: str, output_folder: str, 
                     perturb_type: str = 'none', perturb_level: float = 0.0, save_input: bool = False):
        case_name = os.path.basename(input_path).split('.nii')[0]
        try:
            data, _, properties = self.preprocessor.run_case([input_path], None, self.plans_manager, self.configuration_manager, self.dataset_json)

            if perturb_type != 'none':
                # 基于 case_name 生成唯一的 hash 种子
                import hashlib
                file_hash = int(hashlib.sha256(case_name.encode('utf-8')).hexdigest(), 16) % (10**8)
                data = apply_perturbation(data, perturb_type, perturb_level, seed=file_hash)

            data_tensor = torch.from_numpy(data).to(self.device, non_blocking=True).float()
            
            if save_input and perturb_type != 'none':
                input_save_name = f"{case_name}_input_{perturb_type}_{perturb_level}.nii.gz"
                self._export_tensor_to_nifti(data_tensor.cpu(), properties, os.path.join(output_folder, input_save_name), is_segmentation=False)
            
            evidence = self.predict_logits_from_preprocessed_data(data_tensor).cpu()
            
            alpha = evidence[1:2, ...] + 1.0  # Shape: (1, D, H, W)
            beta = evidence[0:1, ...] + 1.0   # Shape: (1, D, H, W)
            probability_map = alpha / (alpha + beta) # Shape: (1, D, H, W)
            segmentation_mask = (probability_map > 0.5).long() # Shape: (1, D, H, W)
            total_evidence = alpha + beta
            
            print(f"\n正在后处理并保存: {case_name}")
            results_to_save = {
                "seg": (segmentation_mask, True),
                "prob": (probability_map, False),
                # "alpha": (alpha, False),  
                # "beta": (beta, False)     
                # "total_evidence": (total_evidence, False)
            }
            
            
            try:
                # 准备输入，形状为 (2, D, H, W)
                dirichlet_params = torch.cat([beta, alpha], dim=0)

                # 计算基于方差的不确定性
                au_var, eu_var = UncertaintyEvaluation.calculate_uncertainties(dirichlet_params, method='variance')
                # 添加通道维度以便导出，形状 -> (1, D, H, W)
                results_to_save["unc_variance_aleatoric"] = (au_var.unsqueeze(0), False)
                results_to_save["unc_variance_epistemic"] = (eu_var.unsqueeze(0), False)

                # 计算基于熵的不确定性
                au_ent, eu_ent = UncertaintyEvaluation.calculate_uncertainties(dirichlet_params, method='entropy')
                # 添加通道维度以便导出，形状 -> (1, D, H, W)
                # results_to_save["unc_entropy_aleatoric"] = (au_ent.unsqueeze(0), False)
                # results_to_save["unc_entropy_epistemic"] = (eu_ent.unsqueeze(0), False)

                # 计算并保存 Vacuity
                vac_map, _ = UncertaintyEvaluation.calculate_uncertainties(dirichlet_params, method='vacuity')
                results_to_save["unc_vacuity"] = (vac_map.unsqueeze(0), False)
                
            except Exception as e: 
                print(f"计算不确定性时发生严重错误 (已忽略): {e}")
                import traceback
                traceback.print_exc()

            for suffix, (tensor_data, is_seg) in results_to_save.items():
                output_path = os.path.join(output_folder, f"{case_name}_{suffix}.nii.gz")
                self._export_tensor_to_nifti(tensor_data.cpu(), properties, output_path, is_seg)

        except Exception as e:
            print(f"处理病例 {case_name} 时发生严重错误: {e}")
            import traceback; traceback.print_exc()
    
    
    @torch.inference_mode()
    def predict_sliding_window_return_logits(self, input_image: torch.Tensor) -> torch.Tensor:
        assert isinstance(input_image, torch.Tensor)
        self.network = self.network.to(self.device)
        self.network.eval()
        torch.cuda.empty_cache()
        assert input_image.ndim == 4, '输入图像必须是4D张量 (c, x, y, z)'
        if self.verbose: print(f'输入形状: {input_image.shape}')
        data, slicer_revert_padding = pad_nd_image(input_image, self.configuration_manager.patch_size, 'constant', {'value': 0}, True, None)
        slicers = self._internal_get_sliding_window_slicers(data.shape[1:])
        predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers, self.perform_everything_on_device)
        torch.cuda.empty_cache()
        predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]
        return predicted_logits

    @torch.inference_mode()
    def _internal_predict_sliding_window_return_logits(self, data: torch.Tensor, slicers: list, do_on_device: bool = True):
        results_device = self.device if do_on_device else torch.device('cpu')
        predicted_logits = torch.zeros((self.label_manager.num_segmentation_heads, *data.shape[1:]), dtype=torch.float, device=results_device)
        n_predictions = torch.zeros(data.shape[1:], dtype=torch.float, device=results_device)
        if self.use_gaussian:
            gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8, value_scaling_factor=10, device=results_device).float()
        else:
            gaussian = 1.0
        data = data.to(results_device)
        queue = Queue(maxsize=2)
        
        def producer(d, slh, q):
            for s in slh: 
                q.put((d[s][None], s))
            q.put('end')

        producer_thread = Thread(target=producer, args=(data, slicers, queue))
        producer_thread.start()

        with tqdm(desc="滑窗推理", total=len(slicers), disable=not self.allow_tqdm) as pbar:
            while True:
                item = queue.get()
                if item == 'end':
                    queue.task_done()
                    break
                
                patch_data, slicer = item
                patch_data = patch_data.to(self.device, non_blocking=True)
                prediction = self._internal_maybe_mirror_and_predict(patch_data)[0].to(results_device)
                prediction = prediction.squeeze(0).float()
                
                if self.use_gaussian: prediction *= gaussian
                predicted_logits[(slice(None), *slicer[1:])] += prediction
                n_predictions[slicer[1:]] += gaussian
                queue.task_done()
                pbar.update()

        producer_thread.join()
        n_predictions[n_predictions == 0] = 1e-8
        predicted_logits /= n_predictions.unsqueeze(0)
        
        if torch.any(torch.isinf(predicted_logits)) or torch.any(torch.isnan(predicted_logits)):
            raise RuntimeError('在预测数组中遇到inf或nan。推理中止...')
        return predicted_logits     
    
def main():
    """
    推理传统edl模型
    
    CUDA_VISIBLE_DEVICES=3 python my_npc_project/04_inference.py
    """
    parser = argparse.ArgumentParser(description="推理脚本")
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
    default_results_path = os.path.join(default_base_path, 'nnUNet_results', dataset_name, 'EDL_UNet__EDLPlans__3d_fullres__fold0') # 原始模型
    # default_results_path = os.path.join(default_base_path, 'nnUNet_results', dataset_name, 'EDL_UNet_ratio0_08__EDLPlans__3d_fullres__fold0') # 不同比例
    
    parser.add_argument('--model_path', type=str, default=os.path.join(default_results_path, 'best_model.pth'))
    parser.add_argument('--output_folder', type=str, default=os.path.join(default_results_path, 'inference_real_final'))
    parser.add_argument('--input_folder', type=str, default=os.path.join(default_base_path, 'nnUNet_raw', dataset_name, 'imagesTs'))
    parser.add_argument('--plans_file', type=str, default=os.path.join(default_base_path, 'nnUNet_preprocessed', dataset_name, 'EDLPlans.json'))
    parser.add_argument('--dataset_json', type=str, default=os.path.join(default_base_path, 'nnUNet_preprocessed', dataset_name, 'dataset.json'))
    # ----------------------------------------------------

    # [OOD 参数]
    parser.add_argument('--perturb_type', type=str, default='none', choices=['none', 'noise', 'blur'])
    parser.add_argument('--perturb_level', type=float, default=0.0)
    parser.add_argument('--save_perturbed_input', action='store_true')


    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    
    # 全局禁用 SimpleITK 的警告输出
    sitk.ProcessObject.SetGlobalWarningDisplay(False)

    print("--- 1. 初始化 EDL-UNet 预测器 ---")
    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    os.makedirs(args.output_folder, exist_ok=True)
    
    predictor = EDLUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    predictor.initialize_from_paths(
        model_path=args.model_path,
        plans_path=args.plans_file,
        dataset_json_path=args.dataset_json,
        configuration_name='3d_fullres'
    )

    print("\n--- 2. 执行推理 ---")
    test_files = sorted([f for f in os.listdir(args.input_folder) if f.endswith(('.nii.gz', '.nii'))])

    for file_name in tqdm(test_files, desc="总推理进度"):
        image_path = os.path.join(args.input_folder, file_name)
        predictor.predict_case(image_path, args.output_folder, 
                               perturb_type=args.perturb_type, 
                               perturb_level=args.perturb_level, 
                               save_input=args.save_perturbed_input)

    print(f"\n--- 所有影像处理完毕，结果已保存至: {args.output_folder} ---")

if __name__ == "__main__":
    main()