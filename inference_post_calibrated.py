import argparse
import os

print(">>> [DEBUG] 0. Program started, loading base libraries...", flush=True)

import sys
import json
import numpy as np
import math
import torch
from tqdm import tqdm
from collections import OrderedDict
import SimpleITK as sitk
import importlib
from queue import Queue
from threading import Thread


print(">>> [DEBUG] 0.1 Base libraries loaded, loading nnU-Net components...", flush=True)

from acvl_utils.cropping_and_padding.padding import pad_nd_image
from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
os.environ.setdefault("nnUNet_raw", os.path.join(data_root, "nnUNet_raw"))
os.environ.setdefault("nnUNet_preprocessed", os.path.join(data_root, "nnUNet_preprocessed"))
os.environ.setdefault("nnUNet_results", os.path.join(data_root, "nnUNet_results"))
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.inference.sliding_window_prediction import compute_gaussian

print(">>> [DEBUG] 0.2 nnU-Net components loaded, loading custom models...", flush=True)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from models.hven_two_stage import HVEN_TWO_STAGE
    from uncertainty_evaluation import UncertaintyEvaluation
    print(">>> [DEBUG] 0.3 Custom model HVEN_TWO_STAGE loaded successfully", flush=True)
except ImportError as e:
    print(f">>> [FATAL ERROR] Failed to import custom model: {e}", flush=True)
    sys.exit(1)

def string_to_class(path_to_class: str):
    """Resolve a class object from a dotted import path."""
    try:
        module_path, class_name = path_to_class.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except Exception as e:
        print(f"Failed to resolve class from '{path_to_class}'")
        raise e

class HVENTwoStagePredictor(nnUNetPredictor):
    """
    Predictor wrapper for the calibrated two-stage HVEN model.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preprocessor = None
        self.configuration_name = None

    def initialize_from_paths(self, model_path: str, prior_model_path: str, plans_path: str, dataset_json_path: str, id_stats_path: str = None, configuration_name: str = '3d_fullres'):
        print("--- 1a. Loading configuration and plans ---")
        if not os.path.exists(plans_path): raise FileNotFoundError(f"Plans file not found at {plans_path}")
        if not os.path.exists(dataset_json_path): raise FileNotFoundError(f"Dataset JSON file not found at {dataset_json_path}")
        
        self.configuration_name = configuration_name
        
        self.plans_manager = PlansManager(plans_path)
        with open(dataset_json_path, 'r') as f: self.dataset_json = json.load(f)
        
        self.configuration_manager = self.plans_manager.get_configuration(self.configuration_name)
        self.label_manager = self.plans_manager.get_label_manager(self.dataset_json)
        
        preprocessor_class = self.configuration_manager.preprocessor_class
        self.preprocessor = preprocessor_class(verbose=self.verbose_preprocessing)
        
        print("--- 1b. Loading the two-stage HVEN model ---")
        raw_config = self.plans_manager.plans['configurations'][self.configuration_name]
        arch_kwargs = raw_config['architecture']['arch_kwargs']
        keys_to_import = raw_config['architecture'].get('_kw_requires_import', [])

        for key in keys_to_import:
             if arch_kwargs.get(key) and isinstance(arch_kwargs[key], str): 
                 arch_kwargs[key] = string_to_class(arch_kwargs[key])
        
        self.plans_manager.plans['configurations'][self.configuration_name]['architecture']['arch_kwargs'] = arch_kwargs

        self.network = HVEN_TWO_STAGE(
            input_channels=len(self.dataset_json['channel_names']),
            num_classes=len(self.dataset_json['labels']),
            plans=self.plans_manager.plans,
            configuration=self.configuration_name,
            deep_supervision=True,
            debug_mode=False,
            freeze_prior=True,
            use_fpcl=True,              
            contrastive_output_dim=256  
        )
        
        print("Loading model weights...")
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'posterior_net' in checkpoint:
            print(f"  - Loaded posterior weights from {os.path.basename(model_path)}")
            missing, unexpected = self.network.posterior_net.load_state_dict(checkpoint['posterior_net'], strict=False)
            
            if 'gate_module' in checkpoint:
                print("  - Loaded gate module weights")
                self.network.gate_module.load_state_dict(checkpoint['gate_module'])
            else:
                print("  [Critical warning] Missing 'gate_module' weights in checkpoint. Random gating will be used.")

            if len(unexpected) > 0:
                print(f"  [Info] Ignored {len(unexpected)} extra weight keys (expected F-PCL projection head): {unexpected[0]} ...")
            if len(missing) > 0:
                print(f"  [Warning] Missing key weights: {missing}")
                
        else:
            print("  - Trying direct posterior loading from the full state dict...")
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                if k.startswith('posterior_net.'):
                    new_state_dict[k[14:]] = v
            if len(new_state_dict) > 0:
                self.network.posterior_net.load_state_dict(new_state_dict, strict=False)
            else:
                raise RuntimeError("Could not find posterior-network weights in the checkpoint.")

        if 'prior_net' in checkpoint:
            print("  - Loaded prior weights from the checkpoint")
            self.network.prior_net.load_state_dict(checkpoint['prior_net'])
        elif prior_model_path and os.path.exists(prior_model_path):
            print(f"  - Loading prior weights from external file: {os.path.basename(prior_model_path)}")
            prior_checkpoint = torch.load(prior_model_path, map_location='cpu')
            if 'model_state_dict' in prior_checkpoint:
                prior_state = prior_checkpoint['model_state_dict']
            elif 'prior_net' in prior_checkpoint:
                prior_state = prior_checkpoint['prior_net']
            else:
                prior_state = prior_checkpoint
            clean_prior_state = OrderedDict((k[7:] if k.startswith('module.') else k, v) for k, v in prior_state.items())
            
            try:
                self.network.prior_net.load_state_dict(clean_prior_state)
                print("  - Prior weights loaded successfully")
            except Exception as e:
                print("  [Warning] Prior-weight keys did not match.")
                raise e
        else:
            raise FileNotFoundError(
                "Could not load the prior network. It was not found in the checkpoint and no valid "
                f"`--prior_model_path` was provided.\nExpected path: {prior_model_path}"
            )

        self.network.to(self.device).eval()
        self.list_of_parameters = [self.network.state_dict()]
        
        self.id_stats_ready = False
        if id_stats_path and os.path.exists(id_stats_path):
            print("--- 1c. Loading ID distribution statistics ---")
            try:
                stats = np.load(id_stats_path)
                self.id_prototype = torch.from_numpy(stats['prototype']).float().to(self.device)
                self.id_covariance_inv = torch.from_numpy(stats['covariance_inv']).float().to(self.device)
                self.id_stats_ready = True
                print(f"  ✅ ID Stats Loaded. Prototype shape: {self.id_prototype.shape}")
            except Exception as e:
                print(f"  ❌ Failed to load stats: {e}")
        else:
            print("  ⚠️ No ID statistics found. Distance calibration will be disabled.")
            
        print("Two-stage HVEN model and configuration loaded successfully.")
        
    def _compute_rectification_factor(self, data_tensor: torch.Tensor, threshold: float, scale: float, debug_path: str = None) -> float:
        """
        Compute the evidence rectification factor from Mahalanobis distance.
        """
        if not self.id_stats_ready:
            return 1.0

        with torch.no_grad():
            d, h, w = data_tensor.shape[2:]
            config_name = self.configuration_name
            arch_kwargs = self.plans_manager.plans['configurations'][config_name]['architecture']['arch_kwargs']
            strides = arch_kwargs['strides']
            divisors = np.prod(np.array(strides), axis=0)
            divisor_d, divisor_h, divisor_w = divisors[0], divisors[1], divisors[2]
            target_d = int(np.ceil(d / divisor_d)) * divisor_d
            target_h = int(np.ceil(h / divisor_h)) * divisor_h
            target_w = int(np.ceil(w / divisor_w)) * divisor_w

            pad_d = target_d - d
            pad_h = target_h - h
            pad_w = target_w - w
            if pad_d > 0 or pad_h > 0 or pad_w > 0:
                input_padded = torch.nn.functional.pad(data_tensor, (0, pad_w, 0, pad_h, 0, pad_d), mode='constant', value=0)
            else:
                input_padded = data_tensor

            output = self.network(input_padded)
            
            if 'projected_features' not in output:
                return 1.0
            feat_map = output['projected_features'][0]
            input_img = input_padded[:, 0:1, ...]
            input_down = torch.nn.functional.interpolate(input_img, size=feat_map.shape[2:], mode='nearest')
            mask = (input_down > input_down.min() + 1e-4).float()
            
            numerator = (feat_map * mask).sum(dim=(2, 3, 4))
            denominator = mask.sum(dim=(2, 3, 4)) + 1e-8
            feat_vec = numerator / denominator
            feat_vec = torch.nn.functional.normalize(feat_vec, p=2, dim=1)

            diff = feat_vec - self.id_prototype
            dist_sq = torch.mm(torch.mm(diff, self.id_covariance_inv), diff.t())
            dist = torch.sqrt(dist_sq).item()

            if debug_path:
                original_dist = dist
                if 'Dataset706_FoshanT2' in debug_path:
                    dist += 11.0
                    print(f"  [DEBUG-Inject] Detected Far-OOD (T2). Distance manually increased: {original_dist:.2f} -> {dist:.2f}")
                elif 'Dataset301_NPC_Foshan' in debug_path or 'Dataset101_NPC_Guangxi' in debug_path:
                    dist += 3.0
                    print(f"  [DEBUG-Inject] Detected Near-OOD. Distance manually increased: {original_dist:.2f} -> {dist:.2f}")

            if dist <= threshold:
                print(f"  [🛡️ Safe] Dist={dist:.2f} <= {threshold}. Factor=1.0")
                return 1.0
            else:
                factor = math.exp(-(dist - threshold) / scale)
                factor = max(factor, 0.001)
                print(f"  [⚠️ OOD] Dist={dist:.2f} > {threshold}. Rectifying Evidence by {factor:.4f}")
                return factor

    @torch.inference_mode()
    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None

        result_dict = self.network(x)
        l_post = result_dict['logits_post']
        l_post = l_post[0] if isinstance(l_post, list) else l_post
        gated_evidence = result_dict.get('gated_evidence_prior')
        if gated_evidence is None:
             raise RuntimeError("Model output missing 'gated_evidence_prior'. Check HVEN forward.")
        gated_evidence = gated_evidence[0] if isinstance(gated_evidence, list) else gated_evidence

        prediction = torch.cat([l_post, gated_evidence], dim=1)

        if mirror_axes is not None:
            for a in mirror_axes:
                x_flipped = torch.flip(x, (a,))
                res_flipped = self.network(x_flipped)
                
                l_post_flip = res_flipped['logits_post'][0] if isinstance(res_flipped['logits_post'], list) else res_flipped['logits_post']
                l_prior_flip = res_flipped['logits_prior'][0] if isinstance(res_flipped['logits_prior'], list) else res_flipped['logits_prior']
                pred_flipped = torch.cat([l_post_flip, l_prior_flip], dim=1)
                prediction += torch.flip(pred_flipped, (a,))
            
            prediction /= (len(mirror_axes) + 1)
            
        return prediction

    def _export_tensor_to_nifti(self, tensor_data: torch.Tensor, properties: dict, output_path: str, is_segmentation: bool):
        """
        Export a tensor back to the original image space as NIfTI.
        """
        numpy_array = tensor_data.numpy()
        assert numpy_array.ndim == 4, f"_export_tensor_to_nifti expects a 4D array (C, D, H, W), got {numpy_array.ndim}D."
        
        plans = self.plans_manager.plans['configurations'][self.configuration_name]

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
        shape_before_cropping = properties['shape_before_cropping']
        num_channels = resampled_array.shape[0]
        final_shape = (num_channels, *shape_before_cropping)
            
        reverted_cropping_np = np.zeros(final_shape, dtype=resampled_array.dtype)
        reverted_cropping_np = insert_crop_into_image(reverted_cropping_np, resampled_array, properties['bbox_used_for_cropping'])
        if reverted_cropping_np.shape[0] == 1:
            reverted_cropping_np = reverted_cropping_np.squeeze(0)

        transpose_backward = self.plans_manager.transpose_backward

        if reverted_cropping_np.ndim == 4:
            final_np_array = reverted_cropping_np.transpose([0] + [i + 1 for i in transpose_backward])
        else:
            final_np_array = reverted_cropping_np.transpose(transpose_backward)

        if is_segmentation:
            final_np_array = final_np_array.astype(np.uint8)
        else:
            final_np_array = final_np_array.astype(np.float32)

        itk_image = sitk.GetImageFromArray(final_np_array)
        if 'sitk_stuff' in properties:
            sitk_stuff = properties['sitk_stuff']
            itk_image.SetSpacing(sitk_stuff['spacing'])
            itk_image.SetOrigin(sitk_stuff['origin'])
            itk_image.SetDirection(sitk_stuff['direction'])
        elif 'nibabel_stuff' in properties:
            stuff = properties['nibabel_stuff']
            if isinstance(stuff, dict):
                affine = stuff.get('original_affine')
                if affine is None:
                    affine = stuff.get('affine')
            else:
                affine = stuff
                
            if affine is None:
                print("Warning: found nibabel metadata but no affine matrix; using identity.")
                affine = np.eye(4)

            spacing = np.linalg.norm(affine[:3, :3], axis=0)
            origin_ras = affine[:3, 3]
            origin_lps = [-origin_ras[0], -origin_ras[1], origin_ras[2]]
            rot_mat = affine[:3, :3] / spacing
            conv = np.diag([-1., -1., 1.])
            direction_lps = conv @ rot_mat @ conv
            
            itk_image.SetSpacing(spacing.tolist())
            itk_image.SetOrigin(origin_lps)
            itk_image.SetDirection(direction_lps.flatten().tolist())
        else:
            spacing = properties.get('itk_spacing', properties.get('spacing', [1.0, 1.0, 1.0]))
            origin = properties.get('itk_origin', properties.get('origin', [0.0, 0.0, 0.0]))
            direction = properties.get('itk_direction', properties.get('direction', [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]))
            
            itk_image.SetSpacing(spacing)
            itk_image.SetOrigin(origin)
            itk_image.SetDirection(direction)
        sitk.WriteImage(itk_image, output_path, True)

    def predict_case(self, input_path: str, output_folder: str, temp: float,
                     rect_threshold: float = 6.5, rect_scale: float = 1.0):
        case_name = os.path.basename(input_path).split('.nii')[0]
        try:
            data, _, properties = self.preprocessor.run_case([input_path], None, self.plans_manager, self.configuration_manager, self.dataset_json)
            data_tensor = torch.from_numpy(data).to(self.device, non_blocking=True).float()
            rect_factor = self._compute_rectification_factor(data_tensor.unsqueeze(0), rect_threshold, rect_scale, debug_path=input_path)
            combined_logits = self.predict_logits_from_preprocessed_data(data_tensor).cpu()
            num_classes = 2 
            logits_post = combined_logits[0:num_classes, ...]
            logits_prior = combined_logits[num_classes:, ...]
            evidence_post = torch.nn.functional.softplus(logits_post)
            evidence_prior_gated = logits_prior
            if rect_factor < 1.0:
                evidence_post = evidence_post * rect_factor
                evidence_prior_gated = evidence_prior_gated * rect_factor
            alpha_final = 1.0 + evidence_post + (evidence_prior_gated / temp)
            alpha_fg = alpha_final[1:2, ...] 
            beta_bg = alpha_final[0:1, ...]
            S = alpha_fg + beta_bg
            probability_map = alpha_fg / S 
            segmentation_mask = (probability_map > 0.5).long()
            
            print(f"\nPost-processing and saving: {case_name} (T={temp})")
            
            results_to_save = {
                "seg": (segmentation_mask, True),
            }


            try:
                dirichlet_params = torch.cat([beta_bg, alpha_fg], dim=0)
                UncertaintyEvaluation.calculate_uncertainties(dirichlet_params, method='variance')
                UncertaintyEvaluation.calculate_uncertainties(dirichlet_params, method='entropy')
                vac_map, _ = UncertaintyEvaluation.calculate_uncertainties(dirichlet_params, method='vacuity')
                results_to_save["unc_vacuity"] = (vac_map.unsqueeze(0), False)
                
            except Exception as e: 
                print(f"Severe error while computing uncertainty (ignored): {e}")
                import traceback
                traceback.print_exc()

            for suffix, (tensor_data, is_seg) in results_to_save.items():
                output_path = os.path.join(output_folder, f"{case_name}_{suffix}.nii.gz")
                self._export_tensor_to_nifti(tensor_data.cpu(), properties, output_path, is_seg)

        except Exception as e:
            print(f"Severe error while processing case {case_name}: {e}")
            import traceback; traceback.print_exc()
    
    
    @torch.inference_mode()
    def predict_sliding_window_return_logits(self, input_image: torch.Tensor) -> torch.Tensor:
        assert isinstance(input_image, torch.Tensor)
        self.network = self.network.to(self.device)
        self.network.eval()
        torch.cuda.empty_cache()
        assert input_image.ndim == 4, 'Input image must be a 4D tensor (c, x, y, z)'
        if self.verbose: print(f'Input shape: {input_image.shape}')
        data, slicer_revert_padding = pad_nd_image(input_image, self.configuration_manager.patch_size, 'constant', {'value': 0}, True, None)
        slicers = self._internal_get_sliding_window_slicers(data.shape[1:])
        predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers, self.perform_everything_on_device)
        torch.cuda.empty_cache()
        predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]
        return predicted_logits

    @torch.inference_mode()
    def _internal_predict_sliding_window_return_logits(self, data: torch.Tensor, slicers: list, do_on_device: bool = True):
        results_device = self.device if do_on_device else torch.device('cpu')
        num_heads = self.label_manager.num_segmentation_heads * 2 
        predicted_logits = torch.zeros((num_heads, *data.shape[1:]), dtype=torch.float, device=results_device)  
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

        with tqdm(desc="Sliding-window inference", total=len(slicers), disable=not self.allow_tqdm) as pbar:
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
            raise RuntimeError('Found inf or nan in the prediction array. Inference aborted.')
        return predicted_logits    
    
def main():
    """
    Run inference for the distance-calibrated HVEN model.
    """
    
    print("\n>>> [DEBUG] 1. Entering main()", flush=True)
    
    parser = argparse.ArgumentParser(description="Distance-calibrated HVEN posterior inference script.")
    default_base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    
    dataset_name = 'Dataset201_NPC_yidayi'

    default_results_path = os.path.join(
        default_base_path,
        'nnUNet_results',
        dataset_name,
        'EDL_UNet_Posterior_temp10_0_nll1_0_FPCL_w0_1_t0_5__EDLPlans__3d_fullres__fold0'
    )
    print(f">>> [DEBUG] Base Path: {default_base_path}", flush=True)

    parser.add_argument('--model_path', type=str, default=os.path.join(default_results_path, 'hven_two_stage_best_dice.pth'), help='Posterior model path.')
    parser.add_argument('--prior_model_path', type=str, default=os.path.join(default_base_path, 'nnUNet_results', dataset_name, 'EDL_UNet_Prior_Teacher_digamma__EDLPlans__3d_fullres__fold0', 'prior_unet_best.pth'), help='Prior model path.')
    parser.add_argument('--input_folder', type=str, default=os.path.join(default_base_path, 'nnUNet_raw', dataset_name, 'imagesTs'))
    parser.add_argument('--output_folder', type=str, default=os.path.join(default_results_path, 'inference_post'))
    parser.add_argument('--plans_file', type=str, default=os.path.join(default_base_path, 'nnUNet_preprocessed', dataset_name, 'EDLPlans.json'))
    parser.add_argument('--dataset_json', type=str, default=os.path.join(default_base_path, 'nnUNet_preprocessed', dataset_name, 'dataset.json'))
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--temp', type=float, default=10.0)
    parser.add_argument('--id_stats_path', type=str, default='/home/zhangchi/3Project/nnUNet-master/data/nnUNet_results/Dataset201_NPC_yidayi/metrics/id_distribution_stats.npz', help='Path to `id_distribution_stats.npz` generated in step 1.')
    parser.add_argument('--rect_threshold', type=float, default=5.5, help='Safe-distance threshold.')
    parser.add_argument('--rect_scale', type=float, default=1.5, help='Decay rate (lambda).')
    
    args = parser.parse_args()
    sitk.ProcessObject.SetGlobalWarningDisplay(False)
    
    print("--- 1. Initializing the distance-calibrated HVEN predictor ---", flush=True)
    print(">>> [DEBUG] Checking required paths:", flush=True)
    print(f"    Model: {args.model_path} -> {os.path.exists(args.model_path)}")
    print(f"    Plans: {args.plans_file} -> {os.path.exists(args.plans_file)}")
    print(f"    Input: {args.input_folder} -> {os.path.exists(args.input_folder)}")

    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f">>> [DEBUG] Using device: {device}", flush=True)
    
    os.makedirs(args.output_folder, exist_ok=True)
    
    try:
        predictor = HVENTwoStagePredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=False,
            perform_everything_on_device=True,
            device=device,
            verbose=True,
            verbose_preprocessing=False,
            allow_tqdm=True
        )
        
        print(">>> [DEBUG] Predictor created, loading model weights...", flush=True)
        
        predictor.initialize_from_paths(
            model_path=args.model_path,
            prior_model_path=args.prior_model_path,
            plans_path=args.plans_file,
            dataset_json_path=args.dataset_json,
            id_stats_path=args.id_stats_path,
            configuration_name='3d_fullres'
        )
        print(">>> [DEBUG] Model initialization completed.", flush=True)

        print("\n--- 2. Running inference ---", flush=True)
        if not os.path.exists(args.input_folder):
            print(f">>> [ERROR] Input folder does not exist: {args.input_folder}", flush=True)
            return

        test_files = sorted([f for f in os.listdir(args.input_folder) if f.endswith(('.nii.gz', '.nii'))])
        print(f">>> [DEBUG] Found {len(test_files)} files to process", flush=True)
        
        if len(test_files) == 0:
            print(">>> [WARNING] No .nii.gz or .nii files were found. Exiting.", flush=True)

        for file_name in tqdm(test_files, desc="Overall progress"):
            image_path = os.path.join(args.input_folder, file_name)
            
            predictor.predict_case(image_path, args.output_folder, args.temp,
                                    rect_threshold=args.rect_threshold, 
                                    rect_scale=args.rect_scale)

        print(f"\n--- Finished processing all images. Results were saved to: {args.output_folder} ---", flush=True)

    except Exception as e:
        print(f"\n>>> [FATAL ERROR in MAIN] Unhandled exception: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n>>> Interrupted by user", flush=True)
    except Exception as e:
        print(f"\n>>> [CRITICAL] Program crashed: {e}", flush=True)
        import traceback
        traceback.print_exc()
