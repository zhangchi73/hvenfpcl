# Conjugate Bayesian Evidential Learning for Uncertainty-Aware Nasopharyngeal Carcinoma Segmentation

Official code for the paper **Conjugate Bayesian Evidential Learning for Uncertainty-Aware Nasopharyngeal Carcinoma Segmentation**.

This repository provides a nnU-Net-v2-based framework for nasopharyngeal carcinoma (NPC) segmentation with uncertainty estimation.

## Repository Structure

- `train.py`: model training entry point
- `inference.py`: inference and export of segmentation, probability, and uncertainty maps
- `inference_post.py`: HVEN-FPCL posterior inference
- `inference_post_calibrated.py`: HVEN-FPCL inference with FDAEC calibration
- `evaluate_segmentation.py`: segmentation metric evaluation
- `evaluate_calibration.py`: calibration and uncertainty evaluation
- `uncertainty_evaluation.py`: uncertainty computation utilities
- `models/`, `losses/`, `nnunetv2/`: model, loss, and framework-related components
- `data/`: dataset and experiment outputs in nnU-Net-style organization

## Usage

### 1. Environment Setup

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

### 2. Datasets

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

Among the datasets used in this project, only the two public Foshan datasets are publicly available:
[A dataset of primary nasopharyngeal carcinoma MRI with multi-modalities segmentation](https://zenodo.org/records/13131827).
The remaining datasets are private and are not included in this repository.

### 3. Training

Prior network training:

```bash
python train.py -d 201 --use_two_stage_hven --train_prior_only
```

Posterior network training:

```bash
python train.py -d 201 --use_two_stage_hven --freeze_prior --prior_temperature 10.0 --use_fpcl --prior_weights_path data/nnUNet_results/Dataset201_NPC_yidayi/EDL_UNet_Prior_Teacher_digamma__EDLPlans__3d_fullres__fold0/prior_unet_best.pth
```

### 4. Inference

HVEN-FPCL:

```bash
python inference_post.py
```

HVEN-FPCL + FDAEC:

```bash
python inference_post_calibrated.py
```

### 5. Evaluation

Segmentation:

```bash
python evaluate_segmentation.py
```

Uncertainty:

```bash
python evaluate_calibration.py
```

## Citation

Citation information can be added here after the paper is publicly available.

## Contact

For questions or collaboration, please contact: `zhangc31@mail.neu.edu.cn`

## Acknowledgements

We thank the authors and contributors of the following open-source projects:

- [nnUNetv2](https://github.com/MIC-DKFZ/nnUNet)
- [devis](https://github.com/Cocofeat/DEviS)
- [usires](https://github.com/suiannaius/SURE)
