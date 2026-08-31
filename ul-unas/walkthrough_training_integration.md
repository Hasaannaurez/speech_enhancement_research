# ULUNAS Training Pipeline Integration Walkthrough

This document provides a complete walkthrough of all the modifications made to integrate the multi-channel `ULUNAS` model with the `SEtrain` pipeline, using the `src/data` processing logic.

## 1. Defining the Scale-Sensitive Loss (`SEtrain/loss_factory.py`)

We added a new class, `SSNRLoss`, to serve as our optimization objective. 
- **Scale-Sensitivity**: We strictly enforced `alpha = 1` by forcing `y_norm = y_true`. This means the loss doesn't just evaluate the shape/phase of the output, but strictly penalizes any discrepancy in the predicted volume scale versus the target volume scale.
- **Mathematical Scale**: We used `- 20 * torch.log10(...)` to ensure the loss runs on the true mathematical SNR decibel scale, instead of dividing it by 10.

## 2. Training Loop Adjustments (`SEtrain/train.py`)

Several surgical modifications were made to the core training loop to route the new dataset through the model:

- **Cross-Directory Imports**: Added logic to seamlessly import `MixDataset` from `../src/data` and `ULUNAS` from `../ul-unas`.
- **Dataloader Dictionary Unpacking**: The original `DNS3Dataset` yielded a simple tuple `(noisy, clean)`. The new `MixDataset` yields a complex dictionary of components per batch. We updated both the training and validation loops to dynamically pull out:
  - `noisy = batch['noisy_td']` (Multi-channel input for the model)
  - `clean = batch['clean_td']`
- **Reference Microphone Isolation**: `ULUNAS` uses a multi-channel input but outputs a *single-channel* enhanced audio targeting the reference microphone (mic 0). We updated the code to pull `clean = batch['clean_td'][:, 0, :]` so the loss function calculates against a single-channel target.
- **Audio Logging Fix**: Updated `sf.write` in the validation logger. Multi-channel audio in PyTorch is shaped `(Channels, Time)`, but `soundfile` requires `(Time, Channels)`. We appended `.T` to the `noisy` tensor prior to saving so it doesn't crash on validation step 1.

## 3. Configuration Updates (`SEtrain/configs/cfg_train.yaml`)

The training YAML configuration was fully adapted to the new architecture:

- **Network Config**: Swapped out GTCRN parameters for `ULUNAS` arguments, explicitly adding `num_mics: 3`.
- **Loss Config**: Removed the `HybridLoss` tuning weights (`lamda_ri`, `lamda_mag`) and preserved only `eps: 1e-12` for our `SSNRLoss`.
- **Dataset Initialization**: Rewrote the `train_dataset` and `validation_dataset` blocks. 
  - They now pass parameters expected by `MixDataset.__init__`. 
  - Included `meta_frame_length: 48000` (chunking exactly 3 seconds of audio per step since `fs = 16000`).
  - Added placeholders for the `prep_mix` HDF5 file paths, which must be pointed to the actual dataset files on the VM before running.

> **Next Steps on VM**: 
> 1. Pull the `feat/multi_channel_integration_ulunas` branch.
> 2. Open `SEtrain/configs/cfg_train.yaml`.
> 3. Fill in the absolute paths to your generated `prep_mix.hdf5` and `prep_mix_meta.json` files for both the train and validation sets!
> 4. Run `python train.py -C configs/cfg_train.yaml`.
