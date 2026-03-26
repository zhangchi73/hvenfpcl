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

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.inference.sliding_window_prediction import compute_gaussian

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.prior_unet import Prior_UNet
from uncertainty_evaluation import UncertaintyEvaluation

def string_to_class(path_to_class: str):
    """Resolve a class object from a dotted import path."""
    try:
        module_path, class_name = path_to_class.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except Exception as e:
        print(f"Failed to resolve class from '{path_to_class}'")
        raise e

class PriorUNetPredictor(nnUNetPredictor):
    """
    Predictor wrapper for Prior-UNet built on nnU-Net inference utilities.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preprocessor = None
        self.configuration_name = None

    def initialize_from_paths(self, model_path: str, plans_path: str, dataset_json_path: str, configuration_name: str = '3d_fullres'):
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
        
        print("--- 1b. Loading the Prior-UNet model ---")
        arch_kwargs = self.configuration_manager.network_arch_init_kwargs
        for key in self.configuration_manager.network_arch_init_kwargs_req_import:
             if arch_kwargs.get(key): arch_kwargs[key] = string_to_class(arch_kwargs[key])
        
        self.plans_manager.plans['configurations'][self.configuration_name]['architecture']['arch_kwargs'] = arch_kwargs

        self.network = Prior_UNet(
            input_channels=len(self.dataset_json['channel_names']),
            num_classes=2,
            plans=self.plans_manager.plans,
            configuration=self.configuration_name,
            deep_supervision=True,
            debug_mode=False
        )
        
        state_dict = torch.load(model_path, map_location='cpu')
        new_state_dict = OrderedDict((k[7:] if k.startswith('module.') else k, v) for k, v in state_dict.items())
        self.network.load_state_dict(new_state_dict)
        self.network.to(self.device).eval()
        self.list_of_parameters = [self.network.state_dict()]
        print("Model and configuration loaded successfully.")

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
        if 'nibabel_stuff' in properties:
            import nibabel as nib
            affine = properties['nibabel_stuff']['original_affine']
            if final_np_array.ndim == 4:
                final_np_array = final_np_array.transpose(1, 2, 3, 0)
            img = nib.Nifti1Image(final_np_array, affine)
            nib.save(img, output_path)
        else:
            itk_image = sitk.GetImageFromArray(final_np_array)
            
            if 'sitk_stuff' in properties:
                sitk_stuff = properties['sitk_stuff']
                itk_image.SetSpacing(sitk_stuff['spacing'])
                itk_image.SetOrigin(sitk_stuff['origin'])
                itk_image.SetDirection(sitk_stuff['direction'])
            else:
                if 'spacing' in properties: itk_image.SetSpacing(properties['spacing'])
                if 'itk_spacing' in properties: itk_image.SetSpacing(properties['itk_spacing'])
                
                if 'origin' in properties: itk_image.SetOrigin(properties['origin'])
                elif 'itk_origin' in properties: itk_image.SetOrigin(properties['itk_origin'])
                
                if 'direction' in properties: itk_image.SetDirection(properties['direction'])
                elif 'itk_direction' in properties: itk_image.SetDirection(properties['itk_direction'])
            
            sitk.WriteImage(itk_image, output_path, True)

    def predict_case(self, input_path: str, output_folder: str):
        case_name = os.path.basename(input_path).split('.nii')[0]
        try:
            data, _, properties = self.preprocessor.run_case([input_path], None, self.plans_manager, self.configuration_manager, self.dataset_json)
            data_tensor = torch.from_numpy(data).to(self.device, non_blocking=True).float()
            
            evidence = self.predict_logits_from_preprocessed_data(data_tensor).cpu()
            
            alpha = evidence[1:2, ...] + 1.0
            beta = evidence[0:1, ...] + 1.0
            probability_map = alpha / (alpha + beta)
            segmentation_mask = (probability_map > 0.5).long()
            
            print(f"\nPost-processing and saving: {case_name}")
            results_to_save = {
                "seg_prior": (segmentation_mask, True),
                "prob_prior": (probability_map, False),
                "evidence_alpha_prior": (alpha, False),
                "evidence_beta_prior": (beta, False)
            }
            
            
            try:
                dirichlet_params = torch.cat([beta, alpha], dim=0)
                au_var, eu_var = UncertaintyEvaluation.calculate_uncertainties(dirichlet_params, method='variance')
                results_to_save["unc_variance_aleatoric_prior"] = (au_var.unsqueeze(0), False)
                results_to_save["unc_variance_epistemic_prior"] = (eu_var.unsqueeze(0), False)

                au_ent, eu_ent = UncertaintyEvaluation.calculate_uncertainties(dirichlet_params, method='entropy')
                results_to_save["unc_entropy_aleatoric_prior"] = (au_ent.unsqueeze(0), False)
                results_to_save["unc_entropy_epistemic_prior"] = (eu_ent.unsqueeze(0), False)
                
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
    Run Prior-UNet inference to generate evidence maps for the training set.
    """
    parser = argparse.ArgumentParser(description="Prior-UNet inference script for generating evidence maps.")
    default_base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

    dataset_name = 'Dataset201_NPC_yidayi'

    default_results_path = os.path.join(default_base_path, 'nnUNet_results', dataset_name, 'EDL_UNet_Prior_Teacher_digamma__EDLPlans__3d_fullres__fold0')
    
    parser.add_argument('--model_path', type=str, default=os.path.join(default_results_path, 'prior_unet_best.pth'))
    parser.add_argument('--input_folder', type=str, default=os.path.join(default_base_path, 'nnUNet_raw', dataset_name, 'imagesTr'))
    parser.add_argument('--output_folder', type=str, default=os.path.join(default_results_path, 'inference_prior_on_trainset'))
    parser.add_argument('--plans_file', type=str, default=os.path.join(default_base_path, 'nnUNet_preprocessed', dataset_name, 'EDLPlans.json'))
    parser.add_argument('--dataset_json', type=str, default=os.path.join(default_base_path, 'nnUNet_preprocessed', dataset_name, 'dataset.json'))
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    print("--- 1. Initializing the Prior-UNet predictor ---")
    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    os.makedirs(args.output_folder, exist_ok=True)
    
    predictor = PriorUNetPredictor(
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

    print("\n--- 2. Running inference ---")
    test_files = sorted([f for f in os.listdir(args.input_folder) if f.endswith(('.nii.gz', '.nii'))])

    for file_name in tqdm(test_files, desc="Overall progress"):
        image_path = os.path.join(args.input_folder, file_name)
        predictor.predict_case(image_path, args.output_folder)

    print(f"\n--- Finished processing all images. Results were saved to: {args.output_folder} ---")

if __name__ == "__main__":
    main()
