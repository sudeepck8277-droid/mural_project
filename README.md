# MuralDH — AI-Based Digital Mural Restoration

This Flask project combines the segmentation and EDSR models exported by
`MuralDH_Full_3Stage_Training_FlaskReady.ipynb` with pretrained ZeroScratches
detection and Big-LaMa restoration.

## Model files

```text
models/
├── mural_damage_unetplusplus_efficientnetb3.pth
├── big-lama.pt
├── zeroscratches/detection/FT_Epoch_latest.pt
├── final_mural_edsr_x4_dynamic.keras
└── segmentation_metadata.json
```

### Stage 1 — Damage segmentation

PyTorch + `segmentation_models_pytorch`

```python
smp.UnetPlusPlus(
    encoder_name="efficientnet-b3",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None
)
```

The `.pth` checkpoint contains:
- model weights
- encoder name
- input size
- optimal threshold
- ImageNet mean/std

The trained U-Net++ runs alongside the pretrained detector from
[ZeroScratches 1.0.2](https://github.com/leonelhs/zeroscratches). Both receive the
original mural. Their binary masks are aligned to its dimensions and combined
with a union (logical OR): a pixel detected by either model is sent to LaMa.
Overlapping detections count only once. No training or fine-tuning is performed.

Only ZeroScratches' scratch detector is used; LaMa handles all reconstruction.
The adapter uses the library's loaded model with grayscale normalization to
[-1, 1], a 256-pixel short side (long side capped at 1024), and padding to multiples
of 16. It removes padding before aligning the mask to the source image. This
avoids the library's display-mask normalization, which can turn a fully white
mask into a black one. Models load once and inference requires no downloads.

The default scratch threshold is the library's 0.4. To adjust it before startup:

```powershell
$env:MURAL_SCRATCH_THRESHOLD = "0.4"
python app.py
```

Higher values reduce scratch detections; lower values increase coverage and
may classify intact mural details as damage. The union preserves all original
U-Net++ detections. Evaluate the separate masks and restored output before
choosing a threshold; greater mask coverage does not establish better accuracy.

### Stage 2 — Restoration

Pretrained [Big-LaMa](https://github.com/advimman/lama) fills damage using the
surrounding colors, structures and texture. LaMa is an inpainting model: the
combined U-Net++ and ZeroScratches mask guides it. No mural fine-tuning is performed.

- Loads `models/big-lama.pt` once at startup using PyTorch TorchScript.
- Uses CUDA when available, otherwise CPU; no new Python packages are required.
- Takes RGB values normalized to 0–1 and a binary mask (white means damage).
- Excludes a two-pixel rim from the model's context to reduce faded boundary colors.
- Pads to multiples of eight without stretching the image, then removes padding.
- Copies generated pixels back only inside the original damage mask. All other
  pixels in the restored image stay identical; EDSR subsequently enhances the image.
- Processes images up to 1024 pixels on the longest side at native resolution.
  Larger inputs are scaled proportionally for inference and composited at the
  original size. Set `MURAL_LAMA_MAX_INPUT_DIMENSION` to change the limit (minimum 64).
- Fails clearly if weights are missing or no undamaged context remains; it does
  not silently fall back to RGB filling.

The checkpoint is the [simple-lama-inpainting v0.1.0 TorchScript export](https://github.com/enesmsahin/simple-lama-inpainting/releases/tag/v0.1.0).
Its source, size and SHA-256 are recorded in `models/lama_source.json`.
The legacy `best_restoration_autoencoder.keras` is no longer loaded. The earlier
RGB baseline remains available as `RGBColorRestorer` for comparisons.

LaMa estimates missing content; it does not establish the original historical
detail. Damage missed by segmentation remains unchanged. Improvement should be
assessed on representative murals; the project has no clean restoration ground truth.

### Stage 3 — Super Resolution

TensorFlow/Keras dynamic EDSR ×4 model with custom serializable `PixelShuffle`.

---

# 1. Create Anaconda environment

Open **Anaconda Prompt** in this project folder.

```bash
conda env create -f environment.yml
```

Activate:

```bash
conda activate mural_flask
```

Alternative:

```bash
conda create -n mural_flask python=3.10 -y
conda activate mural_flask
pip install -r requirements.txt
```

---

# 2. Copy the trained models

Copy the segmentation checkpoint and EDSR model (plus segmentation metadata, if available) into:

```text
MuralDH_Flask_Full_Project/models/
```

The filenames must match exactly.

Download the pretrained LaMa checkpoint once (about 206 MB):

```bash
python scripts/download_lama.py
python scripts/download_zeroscratches.py
```

ZeroScratches' detector is about 452 MB. Both downloaders verify a pinned SHA-256
and reuse an already verified file. After setup, inference works offline. Both
checkpoints are already in `models/`, and `zeroscratches==1.0.2` is installed
in the `new` Conda environment. Other environments need `pip install -r requirements.txt`.
Scratch model provenance is recorded in `models/zeroscratches_source.json`.

---

# 3. Run Flask

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

# Recommended test input

For a proper unseen-test demonstration use an image from:

```text
Mural_seg/test/images/
```

Do **not** upload the label/mask manually.

The system automatically performs:

```text
Uploaded mural
      ↓
U-Net++ EfficientNet-B3 + ZeroScratches
      ↓
Combined damage mask (union) + overlay
      ↓
Pretrained Big-LaMa inpainting
      ↓
Restored mural
      ↓
Dynamic EDSR ×4
      ↓
High-resolution restored mural
```

---

# UI outputs

The frontend shows:

1. Original image
2. Combined binary damage mask sent to LaMa
3. Separate U-Net++ and ZeroScratches masks
4. Combined damage overlay
5. LaMa restored image
6. Final ×4 enhanced image
7. Combined damage percentage and each detector's contribution
8. U-Net++ and ZeroScratches thresholds and separate probability diagnostics
9. Input and output resolution

---

# Windows GPU note

The segmentation, scratch detection and LaMa models use PyTorch and may use an NVIDIA GPU when a
CUDA-enabled PyTorch build is installed.

TensorFlow 2.16 on native Windows normally uses CPU. The Flask app still works,
but the EDSR stage may be slower. WSL2 is the recommended
modern TensorFlow GPU route on Windows.

For ordinary 512×512 test images, CPU inference is still usable, though EDSR
can take longer.

---

# Troubleshooting

## "Models are not loaded"

Confirm the segmentation checkpoint, `big-lama.pt` and EDSR model exist under
`models/`. Run `python scripts/download_lama.py` if LaMa is missing.
Run `python scripts/download_zeroscratches.py` if the scratch detector is missing.

## Restoration checks

Run the tests in the environment with the app's PyTorch dependencies:

```bash
python -m unittest discover -s tests -v
```

Tests cover RGB channel order, mask polarity, non-square padding, tiny images,
thin damage during resizing, unmasked-pixel preservation and invalid inputs.
They also check union/overlap accounting, independent source masks, scratch-only
damage reaching LaMa, and fully positive/negative scratch predictions.

## PyTorch state-dict mismatch

The segmentation checkpoint must come from the notebook's Flask export:
`mural_damage_unetplusplus_efficientnetb3.pth`.

## PixelShuffle / Keras loading issue

This project registers the exact custom `PixelShuffle` layer used in the
training notebook.

## Very large output / memory use

On CPU, the Flask SR helper limits the EDSR input maximum dimension to 256
pixels before ×4 inference. A 512-pixel EDSR input can consume several GB of
temporary memory and take many minutes on a typical Windows CPU. When
TensorFlow detects a GPU, the default limit is 512 pixels.

You can explicitly choose the limit before starting Flask. Higher values retain
more detail but require substantially more time and memory:

```powershell
$env:MURAL_SR_MAX_INPUT_DIMENSION = "128"
python app.py
```

Use `512` for the previous full-quality behavior on a capable machine.
Use `128` for the fastest response on a CPU-only machine.

## Damage mask seems sparse

The model uses the optimal threshold saved by the notebook in the checkpoint
or `segmentation_metadata.json`. Do not randomly change it before comparing
against validation metrics.
#   m u r a l _ p r o j e c t  
 