# Conjugate Bayesian Evidential Learning for Uncertainty-Aware NPC Segmentation

Official code for the paper **Conjugate Bayesian Evidential Learning for Uncertainty-Aware NPC Segmentation**.

This repository provides a nnU-Net-v2-based framework for nasopharyngeal carcinoma (NPC) segmentation with uncertainty estimation. In addition to segmentation prediction, the code supports evidential learning, uncertainty decomposition, inference export, and quantitative evaluation of both segmentation quality and calibration.

## Highlights

- Uncertainty-aware NPC segmentation based on evidential learning
- Support for probability maps and uncertainty map generation during inference
- Segmentation evaluation with Dice, IoU, HD95, and ASSD
- Calibration and uncertainty evaluation utilities
- Two-stage training pipeline for prior/posterior modeling

## Repository Structure

- `train.py`: model training entry point
- `inference.py`: inference and export of segmentation, probability, and uncertainty maps
- `evaluate_segmentation.py`: segmentation metric evaluation
- `evaluate_calibration.py`: calibration and uncertainty evaluation
- `uncertainty_evaluation.py`: uncertainty computation utilities
- `models/`, `losses/`, `nnunetv2/`: model, loss, and framework-related components
- `data/`: dataset and experiment outputs in nnU-Net-style organization

## Data Organization

Please organize the dataset in nnU-Net format:

```text
data/
  nnUNet_raw/
    Dataset201_NPC_yidayi/
      imagesTr/
      labelsTr/
      imagesTs/
      labelsTs/
      dataset.json
  nnUNet_preprocessed/
  nnUNet_results/
```

The example dataset configuration currently included in this repository is `Dataset201_NPC_yidayi`.

## Environment

Recommended environment:

- Python 3.10
- PyTorch
- SimpleITK
- NiBabel
- SciPy
- pandas
- tqdm
- batchgenerators / batchgeneratorsv2

This project follows the nnU-Net v2 workflow and directory convention.

## Usage

Train:

```bash
python train.py -d 201 -p EDLPlans -c 3d_fullres -f 0
```

Inference:

```bash
python inference.py \
  --model_path data/nnUNet_results/Dataset201_NPC_yidayi/EDL_UNet__EDLPlans__3d_fullres__fold0/best_model.pth \
  --input_folder data/nnUNet_raw/Dataset201_NPC_yidayi/imagesTs \
  --output_folder data/nnUNet_results/Dataset201_NPC_yidayi/EDL_UNet__EDLPlans__3d_fullres__fold0/inference_real_final
```

Segmentation evaluation:

```bash
python evaluate_segmentation.py \
  --pred_folder data/nnUNet_results/Dataset201_NPC_yidayi/EDL_UNet__EDLPlans__3d_fullres__fold0/inference_real_final \
  --gt_folder data/nnUNet_raw/Dataset201_NPC_yidayi/labelsTs
```

Calibration evaluation:

```bash
python evaluate_calibration.py \
  --pred_folder data/nnUNet_results/Dataset201_NPC_yidayi/EDL_UNet__EDLPlans__3d_fullres__fold0/inference_real_final \
  --gt_folder data/nnUNet_raw/Dataset201_NPC_yidayi/labelsTs
```

## Notes

- Medical data are not included in this repository.
- Please prepare your own dataset and preprocessing files before training or inference.
- Citation information can be added here after the paper is publicly available.
