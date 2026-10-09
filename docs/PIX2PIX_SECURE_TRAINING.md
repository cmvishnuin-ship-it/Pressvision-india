# Secure from-scratch pix2pix trainer

**Status:** disabled in the default configuration (`pix2pix_shadow.enabled` in
`src/config.server.json`). Validation PSNR plateaued at about 23 dB while CPU
epochs grew past an hour.

`pix2pix_train.py` is an additional training pipeline. It does not replace or
change the V2 learner, its cache or its checkpoints. When enabled, the service runs
it as a **shadow learner** after V2 and V3. Shadow errors cannot invalidate V2, and
pix2pix checkpoints remain isolated under `artifacts\pix2pix`.

Its architecture follows the paired conditional-GAN principles in Isola et al.,
*Image-to-Image Translation with Conditional Adversarial Networks*:
<https://arxiv.org/abs/1611.07004>. The authors' maintained PyTorch reference is
<https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix>. This implementation is
new local code and does not import pretrained weights or copy a model package.

The trainer learns only from paired images. It contains no production inference
or image-correction command. Original and corrected roots are opened for reads;
checkpoints, logs, and optional staged copies are confined to `--output-dir`.
The safety guard refuses any output or staging path that overlaps a protected
source tree.

## Fixed design

- Native PyTorch U-Net-256 generator with skip connections and Tanh output.
- Native PyTorch conditional 70×70 PatchGAN discriminator.
- No pretrained weights, torchvision model, Hugging Face component, or external
  image-metric package.
- Exact 256×256 RGB resize and `[-1, 1]` normalization for both images.
- Generator objective: adversarial BCE + `100 × L1`.
- Adam learning rate `0.0002`, betas `(0.5, 0.999)` for G and D.
- Native Torch PSNR and Gaussian-window SSIM during every validation epoch.
- Atomic, fsynced checkpoints loaded with Torch's restricted `weights_only` mode.
- CUDA mixed precision, pinned-memory/non-blocking transfers, configurable loader
  workers, dynamic VRAM cache release, and `torchrun` DDP support.

## Pairing behavior

Files are discovered recursively and paired only when their complete filename,
including extension, is identical. An identical relative path is preferred. If
a filename is repeated in corrected subfolders and no identical relative path is
available, the source is rejected as ambiguous; the trainer never guesses.

Automated server runs read `training_cache\cache_index.json` directly. They accept
only local `filename_and_visual` records whose recorded production filenames match
and whose aspect change is no more than 2%. Production paths stored as cache
metadata are never opened, and no second image archive is created.

Replay views from one source share a group identifier and cannot leak across
train/validation. The first automated run freezes its validation groups in
`frozen_validation.json`; later groups enter training, preserving a comparable
benchmark. Those exact validation records are protected from normal cache
compaction so the underlying metric images do not silently change over time.

## Safe validation before training

The following performs path and pair validation only. It does not decode images,
stage data, initialize a model, or train:

```powershell
python pix2pix_train.py `
  --original-root "D:\read-only-example\original" `
  --corrected-root "D:\read-only-example\corrected" `
  --output-dir "C:\ModelTraining\pix2pix-output" `
  --read-only-root "D:\read-only-example" `
  --dry-run
```

## Train

```powershell
python pix2pix_train.py `
  --original-root "D:\read-only-example\original" `
  --corrected-root "D:\read-only-example\corrected" `
  --output-dir "C:\ModelTraining\pix2pix-output" `
  --read-only-root "D:\read-only-example" `
  --epochs 20
```

`--epochs` means additional epochs in the current invocation. Resume only from a
checkpoint inside the selected output's `checkpoints` directory:

```powershell
python pix2pix_train.py <same paths> --epochs 20 --resume last.pt
```

For cache-backed shadow training:

```powershell
python pix2pix_train.py `
  --cache-index "artifacts\training_cache\cache_index.json" `
  --output-dir "artifacts\pix2pix" `
  --epochs 20 --patience 10 --minimum-psnr-delta 0.02 `
  --skip-unchanged --stop-at 09:45 --minimum-remaining-minutes 10 `
  --resume last.pt
```

Omit `--resume last.pt` for the first run. Once early stopping records convergence,
`--skip-unchanged` avoids retraining the same data. New data clears convergence
while retaining the learned weights.

## Optional controlled staging

Staging is off by default. When explicitly enabled, copying may start only inside
the configured 04:00–10:00 local-time window. The gate is checked again before
every pair, files are copied through local `.partial` files, SHA-256 verified,
and atomically installed. Source metadata and source content are never written.
The stage has a non-overridable 300 GB upper bound and never deletes sources.

```powershell
python pix2pix_train.py <same paths> `
  --stage-data `
  --staging-dir "C:\ModelTraining\pix2pix-output\staging" `
  --stage-start 04:00 --stage-end 10:00 `
  --epochs 20
```

Use `--stage-only` to populate/verify the local stage without training.

## Multi-GPU

Launch one process per GPU with native distributed PyTorch:

```powershell
torchrun --standalone --nproc-per-node=2 pix2pix_train.py <same arguments>
```

The rank-zero process alone writes logs and checkpoints. Validation examples are
sharded without duplication and metrics are reduced across ranks.

## Outputs

- `run_metadata.json`: immutable run facts and dataset fingerprint.
- `training_history.jsonl`: one local record per completed epoch.
- `checkpoints/last.pt`: latest atomic checkpoint.
- `checkpoints/best.pt`: highest-validation-PSNR checkpoint.
- `checkpoints/epoch_NNNN.pt`: periodic recovery checkpoint.

Checkpoints store both models, optimizers, AMP scalers, epoch/global-step state,
metrics, fixed hyperparameters, and the data fingerprint. They contain learned
weights, not production images. Periodic recovery files are capped at five by
default; `best.pt` and `last.pt` are never selected by that cleanup rule.

For defense in depth, the Windows/server identity running this script should be
granted only **Read & Execute / List / Read** access on production shares and
write access only on the local model-output location. Application checks cannot
replace operating-system or share ACLs.
