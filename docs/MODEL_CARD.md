# Model card

## Purpose

Learn the corrections a newspaper's photo editors apply in Photoshop (tone, colour,
detail) from pairs of original and edited photos, and measure how much of that
editing a model can reproduce. Resizing and cropping are editorial decisions and
are excluded. The system is learning-only: applying corrections stays disabled
until the measurements are good enough and people have reviewed the results.

## Data

**Pairing** (`correction_ml/pairs.py`):
- Each image gets a visual signature:
  - a 32×32 normalised luminance layout;
  - Sobel edge structure;
  - an 8×8 DCT perceptual hash;
  - 16-bin RGB histograms;
  - 300 ORB features.
- Similarity = 0.38 layout + 0.24 edges + 0.20 hash + 0.08 colour + 0.10 ORB.
- A pair is accepted when:
  - similarity reaches 0.76, or 0.58 if the filenames match;
  - both images are each other's best match, with a 0.045 margin over the
    runner-up;
  - the two files are within 3 days of each other.
- Only pairs whose aspect ratio changed by 2% or less are used for pixel training.

**Measurements** (`correction_ml/metrics.py`): 34 observable effects per pair:
- brightness, contrast and RGB means;
- saturation, colourfulness, warmth and tint;
- hue;
- shadows, mid-tones, highlights and luma percentiles;
- dynamic range, gamma, clipping;
- Laplacian sharpness, high-frequency detail, local contrast, gradients and edge
  density;
- noise and entropy.

These are effects visible in the finished image, not the editors' hidden slider
values, which cannot be recovered uniquely.

**Local cache** (`correction_ml/cache.py`):
- Production folders keep images for only a few days, so verified pairs are
  copied into a local cache.
- Full-resolution pairs stay 30 days (50 GB tier). They are then compacted to one
  512 px whole view plus four lossless 192 px aligned patches, which are verified
  before the full pair is deleted (150 GB replay tier). The overall cap is 300 GB.

**Splits:**
- 15% validation, chosen by a hash of the filename, so a pair never changes side.
- A frozen benchmark set is kept out of training and validation.
- Sampling gives each publication weight proportional to the square root of its
  size, and mixes 60% recent with 40% older pairs.

## Models

**V2: patch model** (`correction_ml/model.py`), 1,994,213 parameters.
- A three-level U-Net with channel-attention blocks (GroupNorm, SiLU).
- The output is residual: input + 0.35·tanh(correction). It starts as an exact
  identity, so an untrained model cannot damage an image.
- A head on the bottleneck predicts the 34 measured effects, which makes the
  model's behaviour explainable.

**V3: whole-picture model** (`correction_ml/model_global.py`), 2,214,403 parameters.
- V2 sees 64 px crops and cannot tell whether the rest of the photo is dark or
  bright. V3 adds an encoder over a 128 px view of the whole image (strided
  convolutions, channel statistics and a 16-bin luminance histogram).
- From that view it predicts, per image:
  - a monotonic 16-knot tone curve per colour channel;
  - a 3×3 colour mix with bias;
  - FiLM modulation of V2's bottleneck and last decoder stage.
- All new layers start at zero, so a model warm-started from V2 reproduces V2
  exactly (checked by the service self-test).
- Only the 220,190 new parameters are trained, so V2's learned detail cannot
  drift.

**Experiments:**
- **Pix2pix:** U-Net-256 generator and 70×70 PatchGAN, 57 M parameters,
  adversarial loss + 100 × L1. It plateaued at about 23 dB PSNR, and a GAN can
  invent detail, so it is disabled.
- **Pretrained variant:** a frozen ImageNet ResNet-18 branch fused at the
  bottleneck.

## Training (`train.py`)

- **Loss:** L1 + 0.15 edge + 0.10 (1 − Gaussian SSIM) + 0.10 colour moments, plus
  0.05 × smooth-L1 on the 34 normalised effects.
  - V3 adds an L1 term comparing its whole-image correction with the editors'
    whole image.
  - Checkpoints are still chosen on picture quality alone.
- **Optimiser:** AdamW, learning rate 2e-4, weight decay 1e-4, gradient clipping
  1.0. The learning rate halves when validation stalls (patience 2), and training
  stops early after 5 epochs without improvement.
- **Samples:** 64 px patches, 4 cut from each decoded image, flips and 90°
  rotations, 2,000 samples per epoch.
- **Validation:** crops are fixed per pair and decoded once per session, so every
  epoch is scored on identical pixels.
- **Resume:** when the data, the loss or the validation method changes, the
  resumed weights are re-scored first. A checkpoint is replaced only by weights
  that beat them.
- **Training data:** only images inside the local cache are used (`--cache-only`);
  pairs that still point at a production path are dropped, never opened.

## Evaluation (`evaluate_whole_images.py`)

Whole held-out images at 768 px, per publication:

- **Reproduced** = 1 − (model error ÷ error of the unchanged image), with mean
  absolute error, PSNR and CIE76 colour difference (ΔE).
- **Pattern check:** the model's average change in brightness, contrast,
  saturation, warmth, tint, shadows, highlights and sharpening, next to the
  editors'.
- **Ceilings:** the best per-image per-channel tone curve and the best per-image
  16³ colour map, both fitted to the editors' own result. They bound what a
  whole-picture correction can reach.

| Correction | Reproduced |
|---|---|
| Unchanged | 0% |
| Fixed preset per publication | 4–10% |
| V2 | 24–30% |
| V3 | 23–29% |
| Ceiling: tone curves | 57% |
| Ceiling: 3D colour map | 66% |

## Limitations

- Finished images do not contain slider values; several settings can produce the
  same pixels.
- JPEG recompression adds to measured pixel and sharpness changes.
- Faces, text, graphics and very dark images need human review before any
  automatic use.
- Training runs on CPU, at low priority, on a shared server.

## Next steps

- Image-adaptive 3D lookup tables (Zeng et al., 2020) to close the gap to the
  66% colour-map ceiling.
- Per-publication sharpening and style (measured from ×2.0 to ×4.3 in fine
  detail).
- Rejecting pairs that are misaligned by more than 1 px at ingest (about 2%).
