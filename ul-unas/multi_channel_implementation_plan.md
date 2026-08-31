# Detailed Implementation Plan: Multi-Channel UL-UNAS (Updated)

This document outlines the exact, minimal changes needed to upgrade UL-UNAS to a multi-channel architecture, strictly following your professor's instructions and how FT-JNF natively handles inputs.

*(Note: Training integration with SEtrain has been omitted from this plan per your request, and will be handled at a later stage).*

## How FT-JNF Handles Inputs
FT-JNF does **not** use power-law compression. It does exactly this:
1. Computes the STFT resulting in a complex tensor.
2. Extracts the raw real and imaginary parts.
3. Concatenates them directly along the channel dimension.

We will mirror this exact logic in our UL-UNAS implementation.

---

## Component-by-Component Breakdown

### 1. Model Initialization (`ul-unas/ulunas.py : ULUNAS.__init__`)
**Changes:**
- Add `num_mics=4` to the `__init__` parameters.
- **Comment out** the ERB initialization: `# self.erb = ERB(...)`.
- **Add Front-End (Replaces BM):** A learnable 2D convolution to fuse the $2 \times C$ raw channels (Real/Imaginary per mic) down to 1 channel, while downsampling the 257 frequency bins to 129 via stride=2.
  ```python
  self.front_end = nn.Conv2d(
      in_channels=2 * num_mics, 
      out_channels=1, 
      kernel_size=(1, 3), # 1 in time (causal), 3 in freq
      stride=(1, 2),      # stride 2 in freq
      padding=(0, 1)      # pad freq to ensure 257 -> 129
  )
  ```
- **Add Back-End (Replaces BS):** A transposed convolution to upsample the 129 bins back to 257, outputting 2 channels for the complex mask.
  ```python
  self.back_end = nn.ConvTranspose2d(
      in_channels=2, 
      out_channels=2, 
      kernel_size=(1, 3), 
      stride=(1, 2), 
      padding=(0, 1), 
      output_padding=(0, 0)
  )
  ```
- **Modify Decoder Output:** Update the last block in `Decoder.__init__` to output `2` channels instead of `1` (for Real/Imaginary components of the cIRM).

### 2. Decoder Final Activation (`ul-unas/ulunas.py : Decoder.forward`)
**Changes:**
- Change `x = torch.sigmoid(x)` to `x = torch.tanh(x)` because a Complex Ideal Ratio Mask (cIRM) must be able to output negative values.

### 3. Forward Pass & Input Processing (`ul-unas/ulunas.py : ULUNAS.forward`)
**Changes:**
- **Input Shape:** Accept `(Batch, C, Time)` instead of `(Batch, Time)`.
- **STFT:** Compute STFT for all $C$ microphones. The output is a complex tensor `X`.
- **Stacking (FT-JNF Style):** We extract the raw real and imaginary parts and stack them natively. 
  ```python
  # X is our complex STFT of shape (B, C, F, T)
  # Concatenate raw real and imaginary parts along the channel dimension
  feat = torch.cat([X.real, X.imag], dim=1)
  ```
- **Permute & Front-End:** Permute `feat` to `(Batch, 2*C, Time, 257)`. Pass this tensor through `self.front_end`. This fuses the $2*C$ channels down to 1, and downsamples 257 bins to 129. Shape becomes `(Batch, 1, Time, 129)`. 
- Note: `# feat = self.erb.bm(feat)` and the old `torch.log10` code will be commented out.

### 4. Encoder & G-DPRNN Bottleneck
**Changes:** **None!**
- Because our Front-End outputs `(Batch, 1, Time, 129)`, it perfectly matches the exact input shape the original Encoder expects. It will process the features completely natively.

### 5. Mask Application (`ul-unas/ulunas.py : ULUNAS.forward`)
**Changes:**
- **Back-End (BS Replacement):** Pass the Decoder's output `(B, 2, T, 129)` through `self.back_end` to upsample it to `(B, 2, T, 257)`. This is our cIRM mask `M`. Note: `# m = self.erb.bs(m_feat)` will be commented out.
- **Complex Multiplication:** 
  - Grab the raw, uncompressed STFT of the **reference microphone** (e.g., mic 0), let's call it $Y_{ref}$.
  - Apply the complex mask: $S_{hat} = (M_{real} + jM_{imag}) \times Y_{ref}$.
- **Inverse STFT:** Pass $S_{hat}$ into `torch.istft` to generate the final 1D enhanced waveform.
