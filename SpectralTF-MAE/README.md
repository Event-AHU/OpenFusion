# SpectralTF

SpectralTF is a Transformer-based framework for tokamak CXRS and XCS spectra. It uses joint peak-aware masked pretraining to learn spectral representations for three downstream tasks.

## Downstream Tasks

1. **Bidirectional Spectral Generation** — Predict CXRS spectra from XCS spectra and XCS spectra from CXRS spectra.
2. **Plasma Parameter Prediction** — Predict electron density (ne) and 32-point ion/electron temperature (Ti/Te) profiles from XCS spectra.
3. **Spectral Peak Decomposition** — Decompose overlapping spectral signals into Gaussian peaks. A pretrained SpectralTF encoder is combined with peak-parameter prediction and bidirectional matching/fusion to estimate peak centers, amplitudes, and full widths at half maximum (FWHM). **The implementation of this task is not publicly released.**

Code for the first two downstream tasks and joint pretraining is included in this repository.

## Dependencies

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

## Run

Input dataset and checkpoint paths are left empty by default. Supply them when running the corresponding scripts.

**Joint pretraining**

```bash
python train_SpectralTF.py -c cfgs/pretrain/spectral_SpectralTF_cxrs_xcs_pretrain.yaml --data_path DATA_DIR
```

**Bidirectional spectral generation**

```bash
python finetune_spectral_xcs_to_cxrs.py -c cfgs/finetune/spectral_xcs_to_cxrs_finetune.yaml --data_path DATA_DIR --pretrained_ckpt PRETRAINED_CKPT
python finetune_spectral_cxrs_to_xcs.py -c cfgs/finetune/spectral_cxrs_to_xcs_finetune.yaml --data_path DATA_DIR --pretrained_ckpt PRETRAINED_CKPT
```

**Plasma parameter prediction**

```bash
python finetune_xcs_ne_tite.py -c cfgs/finetune/spectral_xcs_to_ne_finetune.yaml --data_path DATA_DIR --pretrained_ckpt PRETRAINED_CKPT
```

Replace `DATA_DIR` and `PRETRAINED_CKPT` with your local paths.

## Outputs

The scripts save the best model as `checkpoint_best.pth` and generate task-specific metrics and prediction visualizations under the configured output directory.
