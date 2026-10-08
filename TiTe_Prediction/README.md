
# Ti / Te Profile Prediction

This project predicts tokamak ion temperature (Ti) and electron temperature (Te) profiles using macroscopic parameters and 2D spectral data. The program entry is `TiandTe.py`. A Transformer model is used, and both Ti and Te profiles are uniformly interpolated to 32 radial positions.

## Dependencies

Python 3.6 or above is recommended. Install required packages:

```
pip install numpy pandas torch
```

## Data Preparation

Data should be CSV files placed in the same directory. Please modify the data path in `TiandTe.py` before running.

The CSV must contain the true Ti/Te profiles and spectral data. Main columns:

- `Ti_out`: Ti profile
- `Te_out`: Te profile
- `Spec`: 2D spectral data
- All other non‑excluded columns are used as macroscopic input features

## Run

Automatically select available device:

```
python TiandTe.py
```

Specify GPU (e.g., GPU 0):

```
python TiandTe.py --gpu 0
```

The program automatically performs data filtering and interpolation, train/validation/test splitting, model training, best model saving, test metric calculation, and result plotting.
Default random seed: 42
Default training epochs: 400

## Output Files

- `best_model_attention.pth`: Best model weights on validation set
- `interpolation_raw_vs_smooth_attn.png`: Raw vs. smoothed profiles
- `test_predictions_attn_Ti_soomth.png`: Ti prediction results
- `test_predictions_attn_Te_soomth.png`: Te prediction results
- `test_predictions_attn_RelError.png`: Relative error results

Test set metrics (MSE, RMSE, MAE, R²) will be printed in the terminal.