import argparse
import os
import sys
import torch
import time
import json
import random
import numpy as np
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
os.environ.setdefault("nnUNet_raw", os.path.join(data_root, "nnUNet_raw"))
os.environ.setdefault("nnUNet_preprocessed", os.path.join(data_root, "nnUNet_preprocessed"))
os.environ.setdefault("nnUNet_results", os.path.join(data_root, "nnUNet_results"))


import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.optim import Adam, AdamW
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, load_json, save_json, isfile, subfiles

from models.edl_unet import EDL_UNet
from models.prior_unet import Prior_UNet
from models.hven_two_stage import HVEN_TWO_STAGE
from edl_loss import EvidentialHybridLoss
from losses.two_stage_loss import TwoStageHVENLoss
from losses.fpcl_loss import SelfDistillationLoss
from uncertainty_evaluation import UncertaintyEvaluation

from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results, nnUNet_raw
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2, nnUNetDatasetNumpy, infer_dataset_class
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from nnunetv2.configuration import ANISO_THRESHOLD

from utils.train_tools import (
      HVENDatasetWrapper,
      HVENDataLoader,
      setup_ddp,
      cleanup_ddp,
      string_to_class,
      create_splits,
      count_identifiers,
      perform_sliding_window_validation_prior,
      perform_sliding_window_validation_posterior, 
  )



def main():
    parser = argparse.ArgumentParser(description="Train a 3D U-Net model for the EDL project.")
    parser.add_argument("-d", "--dataset_id", type=int, default=201, help="Dataset ID.")
    parser.add_argument("-p", "--plans", type=str, default="EDLPlans", help="Plans identifier.")
    parser.add_argument("-c", "--config", type=str, default="3d_fullres", help="Configuration to use.")
    parser.add_argument("-f", "--fold", type=int, default=0, help="Fold for cross-validation.")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size to use (overrides plans)")
    parser.add_argument("--create_splits", action="store_true", help="Force creation of new data splits")
    parser.add_argument("--use_fpcl", action="store_true", help="Enable Fuzzy Prototype Contrastive Learning")
    parser.add_argument("--fpcl_loss_weight", type=float, default=0.1, help="Weight for F-PCL contrastive loss")
    parser.add_argument("--fpcl_temperature", type=float, default=0.5, help="Temperature for F-PCL contrastive learning")
    parser.add_argument("--reliability_gamma", type=float, default=1.5, help="Reliability weighting gamma for F-PCL")  
    parser.add_argument("--min_reliability", type=float, default=0.05, help="Minimum reliability weight for F-PCL")
    parser.add_argument("--use_two_stage_hven", action="store_true", help="Enable the new two-stage HVEN architecture with separate Prior-UNet and Posterior-UNet")
    parser.add_argument("--train_prior_only", action="store_true", help="If set, run Stage 1 only and train the Prior-UNet")
    parser.add_argument("--freeze_prior", action="store_true", default=True, help="Freeze the prior network during Stage 2 training")
    parser.add_argument("--prior_weights_path", type=str, default="data/nnUNet_results/Dataset730_NPC/EDL_UNet_Prior_Teacher_digamma__EDLPlans__3d_fullres__fold0/prior_unet_best.pth", help="Path to the prior model weights")
    parser.add_argument("--hven_new_debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--lambda_kl_two_stage", type=float, default=0.01, help="KL divergence weight (deprecated)")
    parser.add_argument("--prior_temperature", type=float, default=10.0, help="Temperature for scaling prior evidence (higher = softer constraint,default: 10.0)")
    parser.add_argument("--kl_annealing_start", type=int, default=50, help="Epoch to start KL annealing in Stage 2 (allows posterior to warm up first)")
    parser.add_argument("--nll_weight", type=float, default=1.0, help="Weight for the NLL term")
    parser.add_argument("--train_subset_ratio", type=float, default=1.0, help="Ratio of training data to use (0.0-1.0, default=1.0 for full dataset)")
    parser.add_argument("--resume", action="store_true", help="Resume training from best_checkpoint.pth if exists")

    args = parser.parse_args()

    if args.use_two_stage_hven:
        if args.train_prior_only:
            print("===== STAGE 1: TRAINING INDEPENDENT PRIOR-UNET (NEW ARCHITECTURE) =====")
            args.two_stage_training = True
            args.training_stage = "prior"
            args.hven_architecture = "two_stage_new"
        else:
            print("===== STAGE 2: TRAINING POSTERIOR-UNET WITH FROZEN PRIOR-UNET (NEW ARCHITECTURE) =====")
            if args.prior_weights_path is None:
                raise ValueError("--prior_weights_path must be provided for Stage 2 training.")
            args.two_stage_training = True
            args.training_stage = "posterior"
            args.hven_architecture = "two_stage_new"
    elif args.train_prior_only:
        raise ValueError("Legacy architecture has been removed. Please use --use_two_stage_hven along with --train_prior_only.")

    is_distributed = setup_ddp()
    
    if is_distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        print(f"Using DDP on device: {device}")
    else:
        local_rank = 0
        device = torch.device("cuda:0")
        print(f"Using single GPU on device: {device}")
    
    seed = 42 + args.fold
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset_name = maybe_convert_to_dataset_name(args.dataset_id)
    
    config_suffix = ""
    if hasattr(args, 'two_stage_training') and args.two_stage_training:
        if args.training_stage == "prior":
            config_suffix = "_Prior_Teacher_digamma"
        elif args.training_stage == "posterior":
            kl_str = str(args.lambda_kl_two_stage).replace('.', '_')
            temp_str = str(args.prior_temperature).replace('.', '_')
            nll_str = str(args.nll_weight).replace('.', '_')
            config_suffix = f"_Posterior_temp{temp_str}_nll{nll_str}"

    if args.use_fpcl:
        fpcl_weight_str = str(args.fpcl_loss_weight).replace('.', '_')
        temp_str = str(args.fpcl_temperature).replace('.', '_')
        config_suffix += f"_FPCL_w{fpcl_weight_str}_t{temp_str}"

    if args.train_subset_ratio < 1.0:
        ratio_str = str(args.train_subset_ratio).replace('.', '_')
        config_suffix += f"_ratio{ratio_str}"
    
    output_dir = join(nnUNet_results, dataset_name, f"EDL_UNet{config_suffix}__{args.plans}__{args.config}__fold{args.fold}")
    if local_rank == 0:
        maybe_mkdir_p(output_dir)
        print(f"Output directory: {output_dir}")

    plans_path = join(nnUNet_preprocessed, dataset_name, args.plans + '.json')
    if not os.path.exists(plans_path):
        if local_rank == 0:
            print(f"Plans file not found at {plans_path}. Please run 02_plan_and_preprocess.py first.")
        return
    plans_manager = PlansManager(plans_path)
    configuration = plans_manager.get_configuration(args.config)
    
    dataset_json_path = join(nnUNet_preprocessed, dataset_name, 'dataset.json')
    if not os.path.exists(dataset_json_path):
        if local_rank == 0:
            print(f"Dataset JSON file not found at {dataset_json_path}. Please run 02_plan_and_preprocess.py first.")
        return
    
    with open(dataset_json_path, 'r') as f:
        dataset_json = json.load(f)
    
    preprocessed_data_folder = join(nnUNet_preprocessed, dataset_name, configuration.data_identifier)
    patch_size = configuration.patch_size
    
    splits_file = join(nnUNet_preprocessed, dataset_name, 'splits_final.json')
    if not isfile(splits_file) or args.create_splits:
        if local_rank == 0:
            print("`splits_final.json` does not exist or a new split was requested.")
            print("Creating a new dataset split...")
            splits = create_splits(nnUNet_preprocessed, dataset_name)
            print(f"Dataset split created and saved to {splits_file}")
        if is_distributed:
            torch.distributed.barrier()
        if not is_distributed or local_rank > 0:
            with open(splits_file, 'r') as f:
                splits = json.load(f)
    else:
        with open(splits_file, 'r') as f:
            splits = json.load(f)
    
    if args.fold >= len(splits):
        if local_rank == 0:
            print(f"Error: fold {args.fold} is out of range. Only {len(splits)} folds available.")
        return
    
    train_identifiers = splits[args.fold]['train']
    val_identifiers = splits[args.fold]['val']
    
    if args.train_subset_ratio < 1.0:
        train_identifiers = sorted(train_identifiers)
        subset_size = int(len(train_identifiers) * args.train_subset_ratio)
        subset_size = max(1, subset_size)
        original_size = len(train_identifiers)
        train_identifiers = train_identifiers[:subset_size]
        if local_rank == 0:
            print(f"[Data subset] Using {args.train_subset_ratio*100:.1f}% of training data: {subset_size} samples (original: {original_size})")
    
    train_size = count_identifiers(train_identifiers)
    val_size = count_identifiers(val_identifiers)
    
    if is_distributed:
        rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        train_identifiers_per_rank = [i for j, i in enumerate(train_identifiers) if j % world_size == rank]
        val_identifiers_per_rank = [i for j, i in enumerate(val_identifiers) if j % world_size == rank]
        train_size_per_rank = count_identifiers(train_identifiers_per_rank)
        val_size_per_rank = count_identifiers(val_identifiers_per_rank)
        
        if local_rank == 0:
            print(f"Process {rank}/{world_size} handles {train_size_per_rank}/{train_size} training samples")
            print(f"Process {rank}/{world_size} handles {val_size_per_rank}/{val_size} validation samples")
    else:
        train_identifiers_per_rank = train_identifiers
        val_identifiers_per_rank = val_identifiers
        train_size_per_rank = train_size
        val_size_per_rank = val_size
        
        if local_rank == 0:
            print(f"Single-GPU training handles {train_size_per_rank} training samples")
            print(f"Single-GPU training handles {val_size_per_rank} validation samples")

    dataset_class = infer_dataset_class(preprocessed_data_folder)

    if (hasattr(args, 'two_stage_training') and args.two_stage_training):
        soft_dataset_id = args.dataset_id + 1
        soft_dataset_name = maybe_convert_to_dataset_name(soft_dataset_id)
        
        preprocessed_np_base = join(nnUNet_preprocessed, soft_dataset_name)
        preprocessed_np_folder = join(preprocessed_np_base, configuration.data_identifier)
        
        if not os.path.exists(preprocessed_np_folder):
            if local_rank == 0:
                print(f"ERROR: Preprocessed NP label folder not found: {preprocessed_np_folder}")
                print("Please ensure soft labels have been preprocessed using:")
                print(f"  python my_npc_project/preprocess_soft_labels.py --dataset_id 730 --num_processes 8")
                print(f"Expected path: {preprocessed_np_folder}")
                print(f"Expected structure: Dataset731_NPC_Prior_soft_preprocessed/{configuration.data_identifier}/")
            return

        base_train_dataset = dataset_class(
            preprocessed_data_folder,
            identifiers=train_identifiers_per_rank
        )
        base_val_dataset = dataset_class(
            preprocessed_data_folder,
            identifiers=val_identifiers_per_rank
        )

        dl_train_dataset = HVENDatasetWrapper(base_train_dataset, preprocessed_np_folder)
        dl_val_dataset = HVENDatasetWrapper(base_val_dataset, preprocessed_np_folder)

        if local_rank == 0:
            if hasattr(args, 'two_stage_training') and args.two_stage_training:
                print("Two-stage training data loading: delayed stacking optimization enabled")
                print(f"   Training stage: {args.training_stage}")
                print(f"   GTV label path: {preprocessed_data_folder}")
                print(f"   NP label path: {preprocessed_np_folder}")
                if args.training_stage == "prior":
                    print("   Supervision target: NP soft labels (channel 2)")
                else:
                    print("   Supervision target: GTV hard labels (channel 1)")
                print("   Optimization: stack labels after `crop_and_pad` so only patches are decompressed")
            else:
                print("HVEN data loading: delayed stacking optimization enabled")
                print(f"   GTV label path: {preprocessed_data_folder}")
                print(f"   NP label path: {preprocessed_np_folder}")
            print("   Expected benefit: 10x-60x faster loading with 90%+ lower memory usage")
    else:
        dl_train_dataset = dataset_class(
            preprocessed_data_folder,
            identifiers=train_identifiers_per_rank
        )
        dl_val_dataset = dataset_class(
            preprocessed_data_folder,
            identifiers=val_identifiers_per_rank
        )
    
    label_manager = plans_manager.get_label_manager(dataset_json)
    
    batch_size = args.batch_size if args.batch_size > 0 else configuration.batch_size

    enable_deep_supervision = False
    if enable_deep_supervision:
        deep_supervision_scales = list(list(i) for i in 1 / np.cumprod(np.vstack(
            configuration.pool_op_kernel_sizes), axis=0))[:-1]
    else:
        deep_supervision_scales = None

    dim = len(patch_size)
    if dim == 2:
        do_dummy_2d_data_aug = False
        if max(patch_size) / min(patch_size) > 1.5:
            rotation_for_DA = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
        else:
            rotation_for_DA = (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)
        mirror_axes = (0, 1)
    elif dim == 3:
        do_dummy_2d_data_aug = (max(patch_size) / patch_size[0]) > ANISO_THRESHOLD
        if do_dummy_2d_data_aug:
            rotation_for_DA = (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)
        else:
            rotation_for_DA = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
        mirror_axes = (0, 1, 2)
    else:
        raise RuntimeError("Unsupported number of dimensions")

    tr_transforms = []
    if do_dummy_2d_data_aug:
        ignore_axes = (0,)
        tr_transforms.append(Convert3DTo2DTransform())
        patch_size_spatial = patch_size[1:]
    else:
        patch_size_spatial = patch_size
        ignore_axes = None

    tr_transforms.append(
        SpatialTransform(
            patch_size_spatial, patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,
            p_rotation=0.2,
            rotation=rotation_for_DA, p_scaling=0.2, scaling=(0.7, 1.4), p_synchronize_scaling_across_axes=1,
            bg_style_seg_sampling=False,
            mode_seg='bilinear'
        )
    )

    if do_dummy_2d_data_aug:
        tr_transforms.append(Convert2DTo3DTransform())

    tr_transforms.append(RandomTransform(
        GaussianNoiseTransform(
            noise_variance=(0, 0.1),
            p_per_channel=1,
            synchronize_channels=True
        ), apply_probability=0.1
    ))
    tr_transforms.append(RandomTransform(
        GaussianBlurTransform(
            blur_sigma=(0.5, 1.),
            synchronize_channels=False,
            synchronize_axes=False,
            p_per_channel=0.5, benchmark=True
        ), apply_probability=0.2
    ))
    tr_transforms.append(RandomTransform(
        MultiplicativeBrightnessTransform(
            multiplier_range=BGContrast((0.75, 1.25)),
            synchronize_channels=False,
            p_per_channel=1
        ), apply_probability=0.15
    ))
    tr_transforms.append(RandomTransform(
        ContrastTransform(
            contrast_range=BGContrast((0.75, 1.25)),
            preserve_range=True,
            synchronize_channels=False,
            p_per_channel=1
        ), apply_probability=0.15
    ))
    tr_transforms.append(RandomTransform(
        SimulateLowResolutionTransform(
            scale=(0.5, 1),
            synchronize_channels=False,
            synchronize_axes=True,
            ignore_axes=ignore_axes,
            allowed_channels=None,
            p_per_channel=0.5
        ), apply_probability=0.25
    ))
    tr_transforms.append(RandomTransform(
        GammaTransform(
            gamma=BGContrast((0.7, 1.5)),
            p_invert_image=1,
            synchronize_channels=False,
            p_per_channel=1,
            p_retain_stats=1
        ), apply_probability=0.1
    ))
    tr_transforms.append(RandomTransform(
        GammaTransform(
            gamma=BGContrast((0.7, 1.5)),
            p_invert_image=0,
            synchronize_channels=False,
            p_per_channel=1,
            p_retain_stats=1
        ), apply_probability=0.3
    ))

    if mirror_axes is not None and len(mirror_axes) > 0:
        tr_transforms.append(
            MirrorTransform(
                allowed_axes=mirror_axes
            )
        )

    tr_transforms.append(RemoveLabelTansform(-1, 0))

    if deep_supervision_scales is not None:
        tr_transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

    tr_transforms_composed = ComposeTransforms(tr_transforms)

    val_transforms = []
    val_transforms.append(RemoveLabelTansform(-1, 0))

    if deep_supervision_scales is not None:
        val_transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

    val_transforms_composed = ComposeTransforms(val_transforms)

    class TransformAdapter:
        def __init__(self, batchgenerators_transform):
            self.transform = batchgenerators_transform

        def __call__(self, **kwargs):
            if 'image' not in kwargs or 'segmentation' not in kwargs:
                raise KeyError(f"Expected 'image' and 'segmentation' keys, but got: {list(kwargs.keys())}")

            try:
                image = kwargs['image']
                segmentation = kwargs['segmentation']

                if not torch.is_tensor(image):
                    image = torch.from_numpy(image)
                if not torch.is_tensor(segmentation):
                    segmentation = torch.from_numpy(segmentation)

                data_dict = {
                    'image': image,
                    'segmentation': segmentation
                }

                transformed = self.transform(**data_dict)

                image_key = 'image' if 'image' in transformed else 'data'
                seg_key = 'segmentation' if 'segmentation' in transformed else 'seg'

                if image_key not in transformed:
                    raise KeyError(f"Expected '{image_key}' key in transform output, got: {list(transformed.keys())}")
                if seg_key not in transformed:
                    raise KeyError(f"Expected '{seg_key}' key in transform output, got: {list(transformed.keys())}")

                result_image = transformed[image_key]
                result_seg = transformed[seg_key]

                if isinstance(result_image, list):
                    result_image = [torch.from_numpy(img) if not torch.is_tensor(img) else img for img in result_image]
                elif not torch.is_tensor(result_image):
                    result_image = torch.from_numpy(result_image)

                if isinstance(result_seg, list):
                    result_seg = [torch.from_numpy(s) if not torch.is_tensor(s) else s for s in result_seg]
                elif not torch.is_tensor(result_seg):
                    result_seg = torch.from_numpy(result_seg)

                # Keep soft labels in a valid probability range after transforms.
                if isinstance(result_seg, list):
                    result_seg = [torch.clamp(s.float(), 0.0, 1.0) for s in result_seg]
                else:
                    result_seg = torch.clamp(result_seg.float(), 0.0, 1.0)

                if isinstance(result_image, list):
                    processed_image = [img.float() for img in result_image]
                else:
                    processed_image = result_image.float()

                if isinstance(result_seg, list):
                    processed_seg = result_seg
                else:
                    processed_seg = result_seg

                result = {
                    'image': processed_image,
                    'segmentation': processed_seg
                }

                return result

            except Exception as e:
                print(f"TransformAdapter error: {e}")
                print(f"Input types: {[(k, type(v)) for k, v in kwargs.items()]}")
                if 'image' in kwargs and hasattr(kwargs['image'], 'shape'):
                    print(f"Image shape: {kwargs['image'].shape}")
                if 'segmentation' in kwargs and hasattr(kwargs['segmentation'], 'shape'):
                    print(f"Segmentation shape: {kwargs['segmentation'].shape}")
                raise

    deterministic_val_transform = ComposeTransforms([RemoveLabelTansform(-1, 0)])

    tr_transforms = TransformAdapter(tr_transforms_composed)
    val_transforms = TransformAdapter(deterministic_val_transform)

    if local_rank == 0:
        print("Deterministic validation is enabled:")
        print(f"  - Training transforms: {len(tr_transforms_composed.transforms)} transforms (including random augmentation)")
        print("  - Validation transforms: deterministic transforms (no randomness)")
        print("  - Validation sampling: fixed random seed to keep patch locations stable")

    if (hasattr(args, 'two_stage_training') and args.two_stage_training):
        DataLoaderClass = HVENDataLoader
        if local_rank == 0:
            if hasattr(args, 'two_stage_training') and args.two_stage_training:
                print("Using HVENDataLoader (two-stage training mode)")
                print(f"   - Training stage: {args.training_stage}")
                if args.training_stage == "prior":
                    print("   - Supervision target: NP soft labels (channel 2)")
                else:
                    print("   - Supervision target: GTV hard labels (channel 1)")
            else:
                print("Using HVENDataLoader (delayed stacking version)")
            print("   - Optimization: only patch regions are decompressed, not full 3D volumes")
            print("   - Expected speedup: 10x-60x faster data loading")
    else:
        DataLoaderClass = nnUNetDataLoader
        if local_rank == 0:
            print("Using the standard nnUNetDataLoader")

    train_loader = DataLoaderClass(
        dl_train_dataset,
        batch_size,
        patch_size,
        patch_size,
        label_manager,
        oversample_foreground_percent=plans_manager.plans['configurations'][args.config].get('oversample_foreground_percent', 0.33),
        sampling_probabilities=None,
        pad_sides=None,
        transforms=tr_transforms
    )

    val_loader = DataLoaderClass(
        dl_val_dataset,
        batch_size,
        patch_size,
        patch_size,
        label_manager,
        oversample_foreground_percent=0,
        sampling_probabilities=None,
        pad_sides=None,
        transforms=None
    )
    num_train_batches_per_epoch = 250
    num_val_batches = 50
    val_size = count_identifiers(val_identifiers)
    
    if local_rank == 0:
        print(f"Training batches per epoch: {num_train_batches_per_epoch}, validation batches: {num_val_batches}")

    if local_rank == 0:
        print("Converting string entries in `plans` to Python classes...")

    arch_params = plans_manager.plans['configurations'][args.config]['architecture']['arch_kwargs']
    keys_to_import = plans_manager.plans['configurations'][args.config]['architecture']['_kw_requires_import']
    
    for key in keys_to_import:
        class_path_string = arch_params.get(key)
        if class_path_string is not None:
            try:
                arch_params[key] = string_to_class(class_path_string)
                if local_rank == 0:
                    print(f"Converted successfully: {key}")
            except Exception as e:
                if local_rank == 0:
                    print(f"Error while converting {key}: {e}")
                dist.destroy_process_group()
                return


    if hasattr(args, 'two_stage_training') and args.two_stage_training:
        if hasattr(args, 'hven_architecture') and args.hven_architecture == "two_stage_new":
            if args.training_stage == "prior":
                model = Prior_UNet(
                    input_channels=len(dataset_json['channel_names']),
                    num_classes=2,
                    plans=plans_manager.plans,
                    configuration=args.config,
                    deep_supervision=True,
                    debug_mode=args.hven_new_debug
                ).to(device)

                if local_rank == 0:
                    print("Using the new Prior-UNet architecture (fully independent)")
                    print("Supervision target: NP soft labels (channel 2)")
                    print("Architecture: fully independent with no shared encoder")

            elif args.training_stage == "posterior":
                model = HVEN_TWO_STAGE(
                    input_channels=len(dataset_json['channel_names']),
                    num_classes=len(dataset_json['labels']),
                    plans=plans_manager.plans,
                    configuration=args.config,
                    deep_supervision=True,
                    debug_mode=args.hven_new_debug,
                    prior_checkpoint=args.prior_weights_path,
                    freeze_prior=args.freeze_prior,
                    use_fpcl=args.use_fpcl
                ).to(device)

                if local_rank == 0:
                    print("Using the new HVEN_TWO_STAGE architecture")
                    print(f"Prior network frozen: {args.freeze_prior}")
                    print("Supervision target: GTV hard labels (channel 1)")
                    print(f"KL constraint weight: {args.lambda_kl_two_stage}")
                    print("Architecture: fully decoupled two-stage design")

            else:
                raise ValueError(f"Unknown training stage: {args.training_stage}")

        else:
            raise ValueError("Legacy two-stage architecture has been removed. Please check your arguments.")

    else:
        model = EDL_UNet(
            input_channels=len(dataset_json['channel_names']),
            num_classes=len(dataset_json['labels']),
            plans=plans_manager.plans,
            configuration=args.config,
            deep_supervision=True
        ).to(device)

        if local_rank == 0:
            print("Using the standard EDL UNet model")
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)


    if hasattr(args, 'two_stage_training') and args.two_stage_training:
        if hasattr(args, 'hven_architecture') and args.hven_architecture == "two_stage_new":
            stage = 1 if args.training_stage == "prior" else 2
            loss_fn = TwoStageHVENLoss(
                stage=stage,
                lambda_kl=args.lambda_kl_two_stage,
                prior_temperature=args.prior_temperature,
                dice_weight=1.0,
                nll_weight=args.nll_weight,
                debug_mode=args.hven_new_debug, 
                kl_annealing_start=args.kl_annealing_start
            )
            if local_rank == 0:
                if stage == 1:
                    print("Using the new TwoStageHVENLoss - stage 1: prior-network training")
                    print("Supervision target: NP soft labels, no KL constraint")
                else:
                    print("Using the new TwoStageHVENLoss - stage 2: posterior-network training")
                    print(f"Supervision target: GTV labels, KL weight: {args.lambda_kl_two_stage}")
        else:
            raise ValueError("Legacy loss function logic has been removed.")

    else:
        kl_annealing_epochs = min(80, args.epochs // 5)
        loss_fn = EvidentialHybridLoss(annealing_epochs=kl_annealing_epochs)

    if args.use_fpcl:
        fpcl_loss_fn = SelfDistillationLoss(
            num_classes=len(dataset_json['labels']),
            temperature=args.fpcl_temperature,
            reliability_gamma=args.reliability_gamma,
            min_reliability=args.min_reliability
        )
        if local_rank == 0:
            print(f"F-PCL self-distillation loss (fixed version): weight={args.fpcl_loss_weight}, temperature={args.fpcl_temperature}")
            print(f"  Reliability gamma={args.reliability_gamma}, minimum reliability={args.min_reliability}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5, betas=(0.9, 0.95))
    
    from torch.optim.lr_scheduler import OneCycleLR

    lr_scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=args.epochs * num_train_batches_per_epoch,
        pct_start=0.3,
        anneal_strategy='cos',
        final_div_factor=1e4
    )

    best_val_loss = float('inf')
    best_val_dice = 0.0
    
    start_epoch = 0
    if args.resume:
        checkpoint_path = join(output_dir, "best_checkpoint.pth")
        if isfile(checkpoint_path):
            if local_rank == 0:
                print(f"Loading checkpoint to resume training: {checkpoint_path}")

            checkpoint = torch.load(checkpoint_path, map_location=device)
            if is_distributed:
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

            start_epoch = checkpoint['epoch']
            if 'best_val_loss' in checkpoint:
                best_val_loss = checkpoint['best_val_loss']
            if 'best_val_dice' in checkpoint:
                best_val_dice = checkpoint['best_val_dice']

            if local_rank == 0:
                print(f"Successfully restored training state. Next epoch: {start_epoch+1}, best loss: {best_val_loss:.4f}, best Dice: {best_val_dice:.4f}")
        else:
            if local_rank == 0:
                print(f"Checkpoint {checkpoint_path} was not found. Training will start from scratch.")
    
    
    if local_rank == 0:
        train_log = []

    data_loading_times = []

    for epoch in range(start_epoch, args.epochs):
        
        model.train()

        start_time = time.time()

        random.shuffle(train_identifiers_per_rank)

        epoch_train_loss = 0
        batch_count = 0

        epoch_train_main_loss = 0.0
        epoch_train_prior_loss = 0.0
        epoch_train_kl_loss = 0.0

        for batch_idx in range(num_train_batches_per_epoch):
            try:
                global_step = epoch * num_train_batches_per_epoch + batch_idx
                current_step = global_step + 1

                data_load_start = time.time()
                batch = train_loader.generate_train_batch()
                data_load_time = time.time() - data_load_start

                if epoch == 0 and batch_idx < 10 and local_rank == 0:
                    data_loading_times.append(data_load_time)
                    print(f"[Performance monitor] Batch {batch_idx}: data loading took {data_load_time:.3f}s")

                if epoch == 0 and batch_idx == 10 and local_rank == 0 and len(data_loading_times) > 0:
                    avg_load_time = sum(data_loading_times) / len(data_loading_times)
                    print(f"\n{'='*80}")
                    print("Data-loading performance summary (average over the first 10 batches):")
                    print(f"   Average loading time: {avg_load_time:.3f}s/batch")
                    print(f"   Fastest: {min(data_loading_times):.3f}s, slowest: {max(data_loading_times):.3f}s")
                    print(f"{'='*80}\n")
                
                if torch.is_tensor(batch['data']):
                    data = batch['data'].to(device)
                else:
                    data = torch.from_numpy(batch['data']).to(device)

                if hasattr(args, 'two_stage_training') and args.two_stage_training:
                    if torch.is_tensor(batch['target']):
                        target_combined = batch['target'].to(device)
                    else:
                        target_combined = torch.from_numpy(batch['target']).to(device)

                    if args.training_stage == "prior":
                        target_np_smooth = target_combined[:, 1:2, ...].float()
                        target_gtv = None
                    elif args.training_stage == "posterior":
                        target_gtv = target_combined[:, 0:1, ...].long()
                        target_np_smooth = None
                    else:
                        raise ValueError(f"Unknown training stage: {args.training_stage}")

                else:
                    if torch.is_tensor(batch['target']):
                        target = batch['target'].to(device)
                    elif isinstance(batch['target'], list):
                        target = [t.to(device) if torch.is_tensor(t) else torch.from_numpy(t).to(device) for t in batch['target']]
                    else:
                        target = torch.from_numpy(batch['target']).to(device)
                
                optimizer.zero_grad()
                
                outputs = model(data)
                
                try:
                    if hasattr(args, 'two_stage_training') and args.two_stage_training:
                        if hasattr(args, 'hven_architecture') and args.hven_architecture == "two_stage_new":
                            if args.training_stage == "prior":
                                model_output = {'logits_prior': outputs}
                                
                                loss_dict = loss_fn(model_output=model_output, target_np=target_np_smooth)
                                total_loss = loss_dict['total_loss']

                            elif args.training_stage == "posterior":
                                loss_dict = loss_fn(model_output=outputs, target_gtv=target_gtv, target_np=target_np_smooth, current_epoch=epoch, total_epochs=args.epochs)
                                hven_loss = loss_dict['total_loss']

                                if args.use_fpcl:
                                    projected_features = outputs.get('projected_features')
                                    if projected_features is None:
                                        raise ValueError("F-PCL enabled but projected_features not found in model output")

                                    logits_post = outputs['logits_post']
                                    l_post_final = logits_post[0] if isinstance(logits_post, list) else logits_post
                                    v_post = F.softplus(l_post_final)

                                    if 'gated_evidence_prior' in outputs:
                                        v_prior_gated = outputs['gated_evidence_prior']
                                        if isinstance(v_prior_gated, list): 
                                            v_prior_gated = v_prior_gated[0]
                                    else:
                                        l_prior = outputs['logits_prior']
                                        l_prior_final = l_prior[0] if isinstance(l_prior, list) else l_prior
                                        v_prior_gated = F.softplus(l_prior_final)

                                    # Build teacher evidence from the fused posterior and gated prior.
                                    T = args.prior_temperature
                                    v_teacher = v_post + (v_prior_gated / T)
                                    alpha_teacher = (v_teacher[:, 1:2, ...] + 1.0).detach()
                                    beta_teacher  = (v_teacher[:, 0:1, ...] + 1.0).detach()

                                    fpcl_outputs = fpcl_loss_fn(
                                        projected_features=projected_features,
                                        evidence_outputs=v_teacher,
                                        alpha_post=alpha_teacher,
                                        beta_post=beta_teacher
                                    )
                                    contrastive_loss = fpcl_outputs['contrastive_loss']

                                    total_loss = hven_loss + args.fpcl_loss_weight * contrastive_loss
                                else:
                                    total_loss = hven_loss

                        else:
                            raise ValueError("Legacy training loop logic has been removed.")
                    
                    elif args.use_fpcl:
                        evidence_outputs = outputs['evidence_outputs']
                        main_loss = loss_fn(evidence_outputs, target, current_epoch=epoch, total_epochs=args.epochs)

                        projected_features = outputs['projected_features']

                        if isinstance(evidence_outputs, list):
                            final_evidence = evidence_outputs[0]
                        else:
                            final_evidence = evidence_outputs

                        alpha_post = final_evidence[:, 1:2, ...] + 1.0
                        beta_post = final_evidence[:, 0:1, ...] + 1.0

                        fpcl_outputs = fpcl_loss_fn(
                            projected_features=projected_features,
                            evidence_outputs=final_evidence,
                            alpha_post=alpha_post,
                            beta_post=beta_post
                        )
                        contrastive_loss = fpcl_outputs['contrastive_loss']

                        total_loss = main_loss + args.fpcl_loss_weight * contrastive_loss

                    else:
                        total_loss = loss_fn(outputs, target, current_epoch=epoch, total_epochs=args.epochs)

                    if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
                        raise ValueError(f"NaN or Inf detected in total_loss at batch {batch_idx}: {total_loss.item()}")

                except (ValueError, RuntimeError, torch.cuda.OutOfMemoryError) as loss_error:
                    if local_rank == 0:
                        print(f"Loss computation error at epoch {epoch+1}, batch {batch_idx}/{num_train_batches_per_epoch}: {loss_error}")
                        print("Skipping this batch and continuing training...")
                        if 'inconsistent tensor size' in str(loss_error):
                            print("  Detected a tensor size mismatch, likely caused by upsampling size issues in the prior decoder")
                            print("  Suggestion: check the `torch.cat` operations in `prior_decoder.py`")
                    torch.cuda.empty_cache()
                    continue

                total_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=12.0)
                
                optimizer.step()
                lr_scheduler.step()

                epoch_train_loss += total_loss.item()
                batch_count += 1

                if batch_idx % 20 == 0:
                    torch.cuda.empty_cache()
                
            except Exception as e:
                if local_rank == 0:
                    print(f"Training-batch processing error at epoch {epoch+1}, batch {batch_idx}/{num_train_batches_per_epoch}:")
                    print(f"  Error type: {type(e).__name__}")
                    print(f"  Error message: {e}")
                    print("  Skipping the batch and continuing training...")
                torch.cuda.empty_cache()
                continue

        if batch_count > 0:
            epoch_train_loss /= batch_count

        model.eval()

        python_random_state = random.getstate()
        numpy_random_state = np.random.get_state()
        torch_random_state = torch.get_rng_state()
        if torch.cuda.is_available():
            torch_cuda_random_state = torch.cuda.get_rng_state()

        # Use a fixed seed so patch sampling stays comparable across epochs.
        validation_seed = 42
        random.seed(validation_seed)
        np.random.seed(validation_seed)
        torch.manual_seed(validation_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(validation_seed)

        epoch_val_loss = 0
        epoch_val_main_loss = 0
        val_batch_count = 0
        
        with torch.no_grad():
            for batch_idx in range(num_val_batches):
                try:
                    batch = val_loader.generate_train_batch()

                    if torch.is_tensor(batch['data']):
                        data = batch['data'].to(device)
                    else:
                        data = torch.from_numpy(batch['data']).to(device)

                    outputs = model(data)

                    if hasattr(args, 'two_stage_training') and args.two_stage_training:
                        if torch.is_tensor(batch['target']):
                            target_combined = batch['target'].to(device)
                        else:
                            target_combined = torch.from_numpy(batch['target']).to(device)

                        if args.training_stage == "prior":
                            target_np_smooth = target_combined[:, 1:2, ...].float()
                            target_gtv = None
                        elif args.training_stage == "posterior":
                            target_gtv = target_combined[:, 0:1, ...].long()
                            target_np_smooth = target_combined[:, 1:2, ...].float()
                    else:
                        if torch.is_tensor(batch['target']):
                            target = batch['target'].to(device)
                        elif isinstance(batch['target'], list):
                            target = [t.to(device) if torch.is_tensor(t) else torch.from_numpy(t).to(device) for t in batch['target']]
                        else:
                            target = torch.from_numpy(batch['target']).to(device)

                    if hasattr(args, 'two_stage_training') and args.two_stage_training:
                        if hasattr(args, 'hven_architecture') and args.hven_architecture == "two_stage_new":
                            if args.training_stage == "prior":
                                model_output = {'logits_prior': outputs}
                                loss_dict = loss_fn(model_output=model_output, target_np=target_np_smooth, current_epoch=epoch, total_epochs=args.epochs)
                                val_loss = loss_dict['total_loss']
                                epoch_val_loss += val_loss.item()
                                epoch_val_main_loss += val_loss.item()
                            elif args.training_stage == "posterior":
                                loss_dict = loss_fn(model_output=outputs, target_gtv=target_gtv, target_np=target_np_smooth, current_epoch=epoch, total_epochs=args.epochs)
                                val_loss = loss_dict['total_loss']
                                epoch_val_loss += val_loss.item()
                                epoch_val_main_loss += loss_dict.get('main_loss', val_loss).item()
                            else:
                                raise ValueError(f"Unknown training stage: {args.training_stage}")
                        else:
                            raise ValueError("Legacy validation logic has been removed.")

                    elif args.use_fpcl:
                        evidence_outputs = outputs['evidence_outputs']
                        val_loss = loss_fn(evidence_outputs, target, current_epoch=epoch, total_epochs=args.epochs)

                        epoch_val_loss += val_loss.item()
                        epoch_val_main_loss += val_loss.item()
                    
                    else:
                        val_loss = loss_fn(outputs, target, current_epoch=epoch, total_epochs=args.epochs)

                        epoch_val_loss += val_loss.item()
                        epoch_val_main_loss += val_loss.item()
                        
                    val_batch_count += 1
                    
                except Exception as e:
                    if local_rank == 0:
                        print(f"Validation-batch generation error: {e}")
                    continue

            if val_batch_count > 0:
                epoch_val_loss /= val_batch_count
                epoch_val_main_loss /= val_batch_count

        random.setstate(python_random_state)
        np.random.set_state(numpy_random_state)
        torch.set_rng_state(torch_random_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(torch_cuda_random_state)

        if is_distributed:
            loss_tensor_full = torch.tensor([epoch_val_loss], device=device)
            dist.all_reduce(loss_tensor_full, op=dist.ReduceOp.AVG)
            avg_val_loss = loss_tensor_full.item()

            loss_tensor_main = torch.tensor([epoch_val_main_loss], device=device)
            dist.all_reduce(loss_tensor_main, op=dist.ReduceOp.AVG)
            avg_val_main_loss = loss_tensor_main.item()
        else:
            avg_val_loss = epoch_val_loss
            avg_val_main_loss = epoch_val_main_loss
        
        end_time = time.time()

        current_lr = optimizer.param_groups[0]['lr']

        if local_rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | Time: {end_time-start_time:.2f}s | "
                    f"Train Loss: {epoch_train_loss:.4f} | Val Loss (Full): {avg_val_loss:.4f} | "
                    f"Val Loss (Main): {avg_val_main_loss:.4f} | LR: {current_lr:.8f}")
            
            if (epoch + 1) % 5 == 0:
                if hasattr(loss_fn, 'last_dice_loss') and loss_fn.last_dice_loss is not None:
                    print(f"--- Loss Magnitudes (Unweighted) ---")
                    print(f"Dice Loss (L_Dice): {loss_fn.last_dice_loss:.4f}")
                    print(f"UCE Loss (L_UCE):   {loss_fn.last_nll_loss:.4f}")

                if hasattr(loss_fn, 'last_evidence_prior') and loss_fn.last_evidence_prior is not None:
                    print(f"--- Evidence Magnitudes [T={loss_fn.prior_temperature}] ---")
                    print(f"Prior Bias (Alpha_Prior):  Mean={loss_fn.last_evidence_prior.mean().item():.4f}, Max={loss_fn.last_evidence_prior.max().item():.4f}")
                    print(f"Update (Alpha_Update):  Mean={loss_fn.last_evidence_update.mean().item():.4f}, Max={loss_fn.last_evidence_update.max().item():.4f}")
                    print(f"Final (Alpha_Posterior): Mean={loss_fn.last_evidence_final.mean().item():.4f}, Max={loss_fn.last_evidence_final.max().item():.4f}")

            log_entry = {
                'epoch': epoch + 1,
                'train_loss': epoch_train_loss,
                'val_loss_full': avg_val_loss,
                'val_loss_main': avg_val_main_loss,
                'lr': current_lr
            }

            if (epoch + 1) % 1 == 0:
                if hasattr(loss_fn, 'last_dice_loss') and loss_fn.last_dice_loss is not None:
                    log_entry['dice_loss_unweighted'] = loss_fn.last_dice_loss
                if hasattr(loss_fn, 'last_nll_loss') and loss_fn.last_nll_loss is not None:
                    log_entry['nll_loss_unweighted'] = loss_fn.last_nll_loss
                
                if hasattr(loss_fn, 'last_evidence_prior') and loss_fn.last_evidence_prior is not None:
                    log_entry['prior_bias_mean'] = loss_fn.last_evidence_prior.mean().item()
                    log_entry['prior_bias_max'] = loss_fn.last_evidence_prior.max().item()
                if hasattr(loss_fn, 'last_evidence_update') and loss_fn.last_evidence_update is not None:
                    log_entry['update_evidence_mean'] = loss_fn.last_evidence_update.mean().item()
                    log_entry['update_evidence_max'] = loss_fn.last_evidence_update.max().item()
                if hasattr(loss_fn, 'last_evidence_final') and loss_fn.last_evidence_final is not None:
                    log_entry['final_alpha_mean'] = loss_fn.last_evidence_final.mean().item()
                    log_entry['final_alpha_max'] = loss_fn.last_evidence_final.max().item()

            train_log.append(log_entry)
            save_json(train_log, join(output_dir, 'training_log.json'), sort_keys=False)

            should_validate = (epoch + 1) % 5 == 0 or epoch == args.epochs - 1
           
            if should_validate and local_rank == 0:
                print(f"\n{'='*80}")
                print(f"Running full validation (sliding-window inference) - Epoch {epoch+1}")
                print(f"{'='*80}")

            if should_validate:
                if hasattr(args, 'two_stage_training') and args.two_stage_training:
                    soft_dataset_id = args.dataset_id + 1
                    soft_dataset_name = maybe_convert_to_dataset_name(soft_dataset_id)
                    preprocessed_np_base = join(nnUNet_preprocessed, soft_dataset_name)
                    
                    true_val_preprocessed_np_folder = join(preprocessed_np_base, configuration.data_identifier)
                else:
                    true_val_preprocessed_np_folder = None

                if hasattr(args, 'hven_architecture') and args.hven_architecture == "two_stage_new":
                    if args.training_stage == "prior":
                        true_val_loss, true_val_dice = perform_sliding_window_validation_prior(
                            model=model.module if is_distributed else model,
                            val_identifiers=val_identifiers,
                            preprocessed_data_folder=preprocessed_data_folder,
                            preprocessed_np_folder=true_val_preprocessed_np_folder,
                            patch_size=patch_size,
                            device=device,
                            loss_fn=loss_fn,
                            dataset_class=dataset_class,
                            local_rank=local_rank,
                            epoch=epoch
                        )
                    else:
                        true_val_loss, true_val_dice = perform_sliding_window_validation_posterior(
                            model=model.module if is_distributed else model,
                            val_identifiers=val_identifiers,
                            preprocessed_data_folder=preprocessed_data_folder,
                            preprocessed_np_folder=true_val_preprocessed_np_folder,
                            patch_size=patch_size,
                            device=device,
                            loss_fn=loss_fn,
                            dataset_class=dataset_class,
                            local_rank=local_rank,
                            epoch=epoch
                        )
                else:
                    raise ValueError("Legacy sliding window validation logic has been removed.")

                if local_rank == 0:
                    print(f"\n{'='*80}")
                    print(f"Full validation results - Epoch {epoch+1}")
                    print(f"{'-'*80}")
                    print(f"  True Loss (Main): {true_val_loss:.4f}")
                    print(f"  True Dice:        {true_val_dice:.4f}")
                    print(f"  (Reference) Online validation loss (main): {avg_val_main_loss:.4f}")
                    print(f"{'-'*80}")

                    train_log[-1]['true_val_loss'] = true_val_loss
                    train_log[-1]['true_val_dice'] = true_val_dice
                    save_json(train_log, join(output_dir, 'training_log.json'), sort_keys=False)

                    should_save = False
                    if hasattr(args, 'two_stage_training') and args.two_stage_training:
                        if args.training_stage == "prior":
                            if true_val_dice > best_val_dice:
                                best_val_dice = true_val_dice
                                should_save = True
                                if local_rank == 0:
                                    print(f"\nFound a better Dice score: {true_val_dice:.4f} (previous: {best_val_dice:.4f})")
                        
                        elif args.training_stage == "posterior":
                            should_save_loss = False
                            if true_val_loss < best_val_loss:
                                best_val_loss = true_val_loss
                                should_save_loss = True
                                if local_rank == 0:
                                    print(f"\nFound a lower loss: {true_val_loss:.4f} (previous: {best_val_loss:.4f})")

                            should_save_dice = False
                            if true_val_dice > best_val_dice:
                                best_val_dice = true_val_dice
                                should_save_dice = True
                                if local_rank == 0:
                                    print(f"\nFound a higher Dice score: {true_val_dice:.4f} (previous: {best_val_dice:.4f})")
                    else:
                        if true_val_loss < best_val_loss:
                            best_val_loss = true_val_loss
                            should_save = True

                    model_state = model.module.state_dict() if is_distributed else model.state_dict()

                    if hasattr(args, 'two_stage_training') and args.two_stage_training and args.training_stage== "prior":
                        if should_save:
                            if hasattr(args, 'hven_architecture') and args.hven_architecture =="two_stage_new":
                                torch.save(model_state, join(output_dir, "prior_unet_best.pth"))
                                if local_rank == 0:
                                    print("\nSaved a new best Prior-UNet model (new architecture)")
                                    print(f"   Epoch:     {epoch+1}")
                                    print(f"   Loss:      {true_val_loss:.4f}")
                                    print(f"   Dice:      {true_val_dice:.4f}")
                                    print(f"   Path:      {join(output_dir, 'prior_unet_best.pth')}")
                            else:
                                raise ValueError("Legacy model saving logic has been removed.")

                            checkpoint = {
                                'epoch': epoch + 1,
                                'model_state_dict': model_state,
                                'optimizer_state_dict': optimizer.state_dict(),
                                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                                'best_val_loss': best_val_loss,
                                'best_val_dice': true_val_dice,
                            }
                            torch.save(checkpoint, join(output_dir, "best_checkpoint.pth"))

                    elif hasattr(args, 'two_stage_training') and args.two_stage_training and args.training_stage == "posterior":
                        if should_save_loss:
                            if hasattr(args, 'hven_architecture') and args.hven_architecture =="two_stage_new":
                                model.save_checkpoint(
                                    filepath=join(output_dir, "hven_two_stage_best.pth"),
                                    epoch=epoch + 1,
                                    best_metric=true_val_loss,
                                    save_posterior=True,
                                    save_prior=False
                                )
                                if local_rank == 0:
                                    print("\nSaved a new best HVEN_TWO_STAGE model (based on loss)")
                                    print(f"   Epoch:     {epoch+1}")
                                    print(f"   Loss:      {true_val_loss:.4f}")
                                    print(f"   Dice:      {true_val_dice:.4f}")
                                    print(f"   Path:      {join(output_dir, 'hven_two_stage_best.pth')}")
                            else:
                                raise ValueError("Legacy model saving logic has been removed.")

                            checkpoint = {
                                'epoch': epoch + 1,
                                'model_state_dict': model_state,
                                'optimizer_state_dict': optimizer.state_dict(),
                                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                                'best_val_loss': best_val_loss,
                                'best_val_dice': best_val_dice,
                            }
                            torch.save(checkpoint, join(output_dir, "best_checkpoint.pth"))

                        if should_save_dice:
                            if hasattr(args, 'hven_architecture') and args.hven_architecture =="two_stage_new":
                                model.save_checkpoint(
                                    filepath=join(output_dir, "hven_two_stage_best_dice.pth"),
                                    epoch=epoch + 1,
                                    best_metric=true_val_dice,
                                    save_posterior=True,
                                    save_prior=False
                                )
                                if local_rank == 0:
                                    print("\nSaved a new best HVEN_TWO_STAGE model (based on Dice)")
                                    print(f"   Epoch:     {epoch+1}")
                                    print(f"   Loss:      {true_val_loss:.4f}")
                                    print(f"   Dice:      {true_val_dice:.4f}")
                                    print(f"   Path:      {join(output_dir, 'hven_two_stage_best_dice.pth')}")
                            else:
                                raise ValueError("Legacy model saving logic has been removed.")

                            checkpoint_dice = {
                                'epoch': epoch + 1,
                                'model_state_dict': model_state,
                                'optimizer_state_dict': optimizer.state_dict(),
                                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                                'best_val_loss': best_val_loss,
                                'best_val_dice': best_val_dice,
                            }
                            torch.save(checkpoint_dice, join(output_dir, "best_checkpoint_dice.pth"))

                    else:
                        if should_save:
                            torch.save(model_state, join(output_dir, "best_model.pth"))
                            if local_rank == 0:
                                print("\nSaved a new best model based on full-validation main loss")
                                print(f"   Epoch:     {epoch+1}")
                                print(f"   Loss:      {true_val_loss:.4f}")
                                print(f"   Dice:      {true_val_dice:.4f}")

                            checkpoint = {
                                'epoch': epoch + 1,
                                'model_state_dict': model_state,
                                'optimizer_state_dict': optimizer.state_dict(),
                                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                                'best_val_loss': best_val_loss,
                                'best_val_dice': true_val_dice,
                            }
                            torch.save(checkpoint, join(output_dir, "best_checkpoint.pth"))

                    print(f"{'='*80}\n")

    if local_rank == 0:
        print("Training complete.")
        print(f"Best validation loss: {best_val_loss:.4f}")

    cleanup_ddp()

if __name__ == "__main__":
    main()
