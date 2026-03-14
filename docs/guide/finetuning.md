# Fine-Tuning Geospatial Foundation Models

This guide provides clear, step-by-step instructions for fine-tuning geospatial foundation models (GFMs) using TerraTorch, along with important considerations to keep in mind.

---

## Overview

Fine-tuning adapts a pre-trained geospatial foundation model to your specific Earth observation task using your own labeled dataset. TerraTorch supports three main task types:

| Task | Class | Use Case |
|------|-------|----------|
| **Semantic Segmentation** | `SemanticSegmentationTask` | Per-pixel class labels (e.g., flood mapping, burn scar detection, land cover) |
| **Classification** | `ClassificationTask` | Scene-level labels (e.g., land use, crop type) |
| **Pixel-wise Regression** | `PixelwiseRegressionTask` | Per-pixel continuous values (e.g., biomass, canopy height, soil moisture) |

---

## Step 1: Install TerraTorch

Set up a virtual environment and install TerraTorch:

```shell
python -m venv venv  # Python 3.10 or newer required
source venv/bin/activate
pip install --upgrade pip
pip install terratorch
```

GDAL is required for reading and writing GeoTIFF files. On Linux this is typically available via your package manager. Otherwise:

```shell
conda install -c conda-forge gdal
```

For weather/climate models (WxC), use Python >= 3.11 and install the extra dependencies:

```shell
pip install "terratorch[wxc]"
```

---

## Step 2: Understand Available Models

TerraTorch provides access to several pre-trained geospatial foundation models. You can list all available backbones:

```python
from terratorch import BACKBONE_REGISTRY

# List all available models
print(list(BACKBONE_REGISTRY))

# Filter by model family
print([m for m in BACKBONE_REGISTRY if "prithvi" in m])
print([m for m in BACKBONE_REGISTRY if "terramind" in m])
print([m for m in BACKBONE_REGISTRY if "clay" in m])
```

Key model families available:

| Model | Variants | Notes |
|-------|----------|-------|
| **Prithvi EO** | `prithvi_eo_v2_300`, `prithvi_eo_v2_600`, `prithvi_eo_v2_300_tl`, `prithvi_eo_v2_600_tl` | IBM-NASA; `_tl` variants support time/location metadata |
| **TerraMind** | `terramind_v1_base`, `terramind_v1_tiny` | ESA foundation model |
| **Clay** | `clay_v1`, `clay_v15` | Open-source EO foundation model |
| **SatMAE** | `satmae_vit_large_patch16` | Masked autoencoder for satellite imagery |
| **ScaleMAE** | `scalemae_vitlarge_window16` | Scale-aware MAE |
| **DOFA** | `dofa_vit_base_patch16`, `dofa_vit_large_patch16` | Multi-spectral foundation model |
| **TIMM** | Any model in [timm](https://github.com/huggingface/pytorch-image-models) | Standard vision backbones via `timm_<model_name>` |

---

## Step 3: Prepare Your Data

### Data Format

TerraTorch's generic datamodules load multi-band GeoTIFF files using `rioxarray`. Verify your files load correctly:

```python
import rioxarray as rxr
rxr.open_rasterio('your_image.tif')
```

Organize your dataset into training, validation, and test splits:

```
my_dataset/
├── training/
│   ├── image_001.tif
│   ├── image_001.mask.tif
│   └── ...
├── validation/
│   ├── image_101.tif
│   ├── image_101.mask.tif
│   └── ...
└── test/
    ├── image_201.tif
    └── ...
```

### Compute Band Statistics

Compute per-band mean and standard deviation for normalization:

```python
import numpy as np
import rioxarray as rxr
from pathlib import Path

files = list(Path("my_dataset/training").glob("*.tif"))
all_data = [rxr.open_rasterio(f).values for f in files if "mask" not in str(f)]
stack = np.stack(all_data, axis=0)  # (N, C, H, W)

means = stack.mean(axis=(0, 2, 3)).tolist()
stds = stack.std(axis=(0, 2, 3)).tolist()
print("means:", means)
print("stds:", stds)
```

---

## Step 4: Choose Your Architecture

TerraTorch uses an **encoder–neck–decoder–head** pipeline, built by the `EncoderDecoderFactory`.

### Select a Backbone (Encoder)

Choose a backbone based on your available spectral bands and compute budget:

- **Prithvi EO v2 300M / 600M**: Strong general-purpose EO backbone; pretrained on Blue, Green, Red, NIR, SWIR-1, SWIR-2.
- **TerraMind**: Good multi-modal performance; use `terramind_v1_base` for best accuracy or `terramind_v1_tiny` for lightweight deployments.
- **Clay v1 / v1.5**: Good for diverse sensor types.
- **TIMM models**: When you need a well-known CNN or ViT architecture.

### Select a Decoder

Common decoders included with TerraTorch:

| Decoder | Best For | Notes |
|---------|----------|-------|
| `UNetDecoder` | Segmentation, pixel-wise regression | Requires pyramidal neck inputs |
| `UperNetDecoder` | Segmentation | Robust, commonly used with ViT backbones |
| `IdentityDecoder` | Feature extraction, embedding generation | Passes features through without change |
| SMP decoders (e.g. `smp_Unet`) | Segmentation | From `segmentation_models_pytorch` |

### Select Necks

Necks bridge the backbone output format to the decoder input format. For ViT-based backbones (e.g., Prithvi, TerraMind):

```yaml
necks:
  - name: SelectIndices
    indices: [5, 11, 17, 23]   # For 300M model (24-layer ViT)
    # indices: [7, 15, 23, 31] # For 600M model (32-layer ViT)
    # indices: [2, 5, 8, 11]   # For tiny/100M models (12-layer ViT)
  - name: ReshapeTokensToImage   # Reshape 1D token sequence → 2D feature map
  - name: LearnedInterpolateToPyramidal  # Scale to pyramid for UNet-style decoders
```

---

## Step 5: Write a Configuration File

The recommended workflow uses a YAML configuration file with the TerraTorch CLI. Here is a complete example for semantic segmentation:

```yaml title="segmentation_config.yaml"
# Reproducibility
seed_everything: 42

trainer:
  accelerator: auto          # Automatically selects GPU if available
  strategy: auto
  devices: auto
  num_nodes: 1
  precision: 16-mixed        # Mixed precision speeds up training
  max_epochs: 50
  check_val_every_n_epoch: 1
  log_every_n_steps: 10
  enable_checkpointing: true
  default_root_dir: output/my_experiment
  logger:
    class_path: TensorBoardLogger
    init_args:
      save_dir: output/my_experiment/logs
  callbacks:
    - class_path: RichProgressBar
    - class_path: LearningRateMonitor
      init_args:
        logging_interval: epoch
    - class_path: EarlyStopping
      init_args:
        monitor: val/loss
        patience: 10

data:
  class_path: GenericNonGeoSegmentationDataModule
  init_args:
    batch_size: 8
    num_workers: 4
    train_data_root: my_dataset/training
    val_data_root: my_dataset/validation
    test_data_root: my_dataset/test
    img_grep: "*.tif"            # Pattern to match image files
    label_grep: "*.mask.tif"     # Pattern to match label files
    dataset_bands:               # Bands present in your image files
      - BLUE
      - GREEN
      - RED
      - NIR_NARROW
      - SWIR_1
      - SWIR_2
    output_bands:                # Bands to pass to the model
      - BLUE
      - GREEN
      - RED
      - NIR_NARROW
      - SWIR_1
      - SWIR_2
    rgb_indices: [2, 1, 0]       # Band indices for RGB visualization
    num_classes: 2
    means: [0.033, 0.057, 0.059, 0.232, 0.197, 0.119]  # Per-band means
    stds:  [0.023, 0.027, 0.040, 0.078, 0.087, 0.072]  # Per-band stds
    no_data_replace: 0           # Replace no-data pixels in images
    no_label_replace: -1         # Replace no-data pixels in labels
    train_transform:
      - class_path: albumentations.D4           # Random 90° rotations + flips
      - class_path: albumentations.pytorch.ToTensorV2
    test_transform:
      - class_path: albumentations.pytorch.ToTensorV2

model:
  class_path: terratorch.tasks.SemanticSegmentationTask
  init_args:
    model_factory: EncoderDecoderFactory
    model_args:
      backbone: prithvi_eo_v2_300
      backbone_pretrained: true
      backbone_num_frames: 1
      backbone_bands:
        - BLUE
        - GREEN
        - RED
        - NIR_NARROW
        - SWIR_1
        - SWIR_2
      necks:
        - name: SelectIndices
          indices: [5, 11, 17, 23]
        - name: ReshapeTokensToImage
        - name: LearnedInterpolateToPyramidal
      decoder: UNetDecoder
      decoder_channels: [512, 256, 128, 64]
      head_dropout: 0.1
      num_classes: 2
    loss: dice                   # Or "ce" for cross-entropy
    ignore_index: -1             # Ignore no-data label pixels during training
    freeze_backbone: false       # false = full fine-tuning; true = linear probing

optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: 6.0e-5
    weight_decay: 0.05

lr_scheduler:
  class_path: ReduceLROnPlateau
  init_args:
    monitor: val/loss
    factor: 0.5
    patience: 5
```

For **classification** or **pixel-wise regression**, replace the task and datamodule:

=== "Classification"
    ```yaml
    data:
      class_path: GenericNonGeoClassificationDataModule
      init_args:
        ...
    model:
      class_path: terratorch.tasks.ClassificationTask
      init_args:
        ...
        loss: ce
    ```

=== "Pixel-wise Regression"
    ```yaml
    data:
      class_path: GenericNonGeoPixelwiseRegressionDataModule
      init_args:
        ...
    model:
      class_path: terratorch.tasks.PixelwiseRegressionTask
      init_args:
        ...
        loss: rmse
    ```

---

## Step 6: Run Fine-Tuning

### Using the CLI (Recommended)

```shell
terratorch fit --config segmentation_config.yaml
```

Monitor training with TensorBoard:

```shell
tensorboard --logdir output/my_experiment/logs
```

### Using the Python API

```python
from lightning.pytorch import Trainer
import albumentations
from albumentations.pytorch import ToTensorV2
import terratorch
from terratorch.datamodules import GenericNonGeoSegmentationDataModule
from terratorch.tasks import SemanticSegmentationTask

datamodule = GenericNonGeoSegmentationDataModule(
    batch_size=8,
    num_workers=4,
    dataset_bands=["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"],
    output_bands=["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"],
    rgb_indices=[2, 1, 0],
    means=[0.033, 0.057, 0.059, 0.232, 0.197, 0.119],
    stds=[0.023, 0.027, 0.040, 0.078, 0.087, 0.072],
    train_data_root="my_dataset/training",
    val_data_root="my_dataset/validation",
    test_data_root="my_dataset/test",
    img_grep="*.tif",
    label_grep="*.mask.tif",
    num_classes=2,
    no_data_replace=0,
    no_label_replace=-1,
    train_transform=[albumentations.D4(), ToTensorV2()],
    test_transform=[ToTensorV2()],
)

model_args = dict(
    backbone="prithvi_eo_v2_300",
    backbone_pretrained=True,
    backbone_num_frames=1,
    backbone_bands=["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"],
    necks=[
        {"name": "SelectIndices", "indices": [5, 11, 17, 23]},
        {"name": "ReshapeTokensToImage"},
        {"name": "LearnedInterpolateToPyramidal"},
    ],
    decoder="UNetDecoder",
    decoder_channels=[512, 256, 128, 64],
    head_dropout=0.1,
    num_classes=2,
)

task = SemanticSegmentationTask(
    model_args=model_args,
    model_factory="EncoderDecoderFactory",
    loss="dice",
    lr=6e-5,
    ignore_index=-1,
    optimizer="AdamW",
    optimizer_hparams={"weight_decay": 0.05},
    freeze_backbone=False,
)

trainer = Trainer(
    accelerator="auto",
    max_epochs=50,
    precision="16-mixed",
)

trainer.fit(model=task, datamodule=datamodule)
```

---

## Step 7: Evaluate and Export

### Evaluate on the Test Set

```shell
terratorch test --config segmentation_config.yaml --ckpt_path output/my_experiment/checkpoints/last.ckpt
```

### Run Inference on New Images

```shell
terratorch predict \
  --config segmentation_config.yaml \
  --ckpt_path output/my_experiment/checkpoints/best.ckpt \
  --predict_output_dir output/predictions \
  --data.init_args.predict_data_root path/to/new/images \
  --data.init_args.predict_dataset_bands "[BLUE,GREEN,RED,NIR_NARROW,SWIR_1,SWIR_2]"
```

---

## Key Considerations

### 1. Band Alignment with Pre-training

Most geospatial foundation models are pre-trained on a fixed set of spectral bands (e.g., Prithvi EO is pre-trained on Blue, Green, Red, NIR Narrow, SWIR-1, SWIR-2). Align your input bands to the pre-training bands where possible:

- **Use the same bands**: Best transfer learning performance.
- **Use a subset**: Pass only the bands available in your data (e.g., RGB only). The model will adapt, but performance may be reduced.
- **Add new bands**: Unknown bands will have their patch embeddings randomly initialized. Use a lower learning rate for those embeddings.

Always set `backbone_bands` explicitly to match what is in your data files.

### 2. Full Fine-Tuning vs. Linear Probing

| Strategy | `freeze_backbone` | When to Use |
|----------|------------------|-------------|
| **Full fine-tuning** | `false` | Sufficient labeled data (hundreds to thousands of samples); best accuracy |
| **Linear probing** | `true` | Very small labeled dataset; fast iteration; use as a baseline |
| **Partial fine-tuning** | — | Fine-tune only the last N backbone layers using `freeze_backbone: true` and manual unfreezing |

In most geospatial scenarios with moderate data (>500 labeled samples), **full fine-tuning** achieves better results than linear probing.

### 3. Learning Rate

The learning rate is one of the most critical hyperparameters:

- **Start with `1e-4` to `1e-5`** for AdamW with full fine-tuning.
- **Use a learning rate scheduler** such as `ReduceLROnPlateau` to reduce the LR when validation loss stagnates.
- Consider **layer-wise learning rate decay**: use a smaller LR for backbone layers and a larger LR for the decoder/head.
- If training diverges or produces NaN losses, try a lower LR or switch from `16-mixed` to full precision (`32`).

### 4. Image Size and Patch Size

- For ViT-based backbones, the image size should ideally be a **multiple of the patch size** (typically 16 pixels). For example: 224×224, 448×448, 512×512.
- TerraTorch applies automatic padding so any size works, but matched sizes yield the cleanest feature maps.
- During inference on large satellite tiles, use **tiled inference** to avoid memory issues:

    ```yaml
    model:
      init_args:
        tiled_inference_parameters:
          h_crop: 224
          h_stride: 192
          w_crop: 224
          w_stride: 192
          average_patches: true
    ```

### 5. Data Augmentation

Augmentation is essential to prevent overfitting, especially with small datasets:

- **Always use**: `albumentations.D4()` (random 90° rotations and flips) — free, label-safe, and effective.
- **For regression tasks**: Avoid color jitter transforms on label images; ensure the same transform is applied to image and label.
- Apply augmentation only to the training set; use only `ToTensorV2()` for validation and test sets.

### 6. Handling Class Imbalance

For segmentation tasks with rare classes (e.g., burn scars, flooded areas):

- Use **Dice loss** (`loss: dice`) or **Focal loss** instead of standard cross-entropy. Dice loss is class-frequency agnostic.
- Use **weighted cross-entropy** by passing `class_weights` to the task.
- Monitor per-class metrics (e.g., Jaccard Index per class) to catch poor performance on minority classes.

### 7. No-data and Missing Values

Geospatial data often contains no-data regions (e.g., cloud masks, scan gaps):

- Set `no_data_replace: 0` in the datamodule to replace no-data pixels in images with zero (neutral after normalization).
- Set `no_label_replace: -1` to mark no-data label pixels, and set `ignore_index: -1` in the task so these pixels are excluded from loss and metric computation.
- If training stops after one epoch with no error, NaN losses are likely the cause. Inspect your data for NaN values or try `precision: 32`.

### 8. Hardware and Batch Size

- **GPU memory**: Larger backbones (600M) require more memory. If you hit OOM errors, reduce `batch_size`, use `precision: 16-mixed`, or reduce image chip size.
- **Recommended chip size**: 224×224 to 512×512 pixels. Larger chips capture more spatial context but consume more memory.
- **Gradient accumulation**: To simulate a larger effective batch size without more memory:

    ```yaml
    trainer:
      accumulate_grad_batches: 4  # Effective batch size = batch_size * 4
    ```

### 9. Preventing Overfitting

- Use **EarlyStopping** with a reasonable patience (e.g., 10 epochs) monitoring `val/loss`.
- Use **weight decay** (`weight_decay: 0.05` with AdamW) for regularization.
- Use **dropout** in the head (`head_dropout: 0.1` to `0.3`).
- Use data augmentation (see above).
- Keep a held-out test set that is never used for hyperparameter selection.

### 10. Multi-temporal Data

For tasks involving time series of imagery (e.g., crop type mapping):

- Set `backbone_num_frames` to the number of time steps (e.g., `3` for three acquisitions).
- For Prithvi `_tl` models, you can optionally pass temporal and location coordinates to improve performance. See the [Prithvi EO guide](prithvi_eo.md#metadata-inputs) for details.
- Alternatively, use the [Temporal Wrapper](temporal_wrapper.md) to apply a model trained on single images to multi-temporal sequences.

### 11. Checkpoint Management

- TerraTorch automatically saves checkpoints when `enable_checkpointing: true` is set.
- By default, the last and best (by validation metric) checkpoints are saved in `default_root_dir/`.
- To resume training from a checkpoint:

    ```shell
    terratorch fit --config config.yaml --ckpt_path path/to/checkpoint.ckpt
    ```

### 12. Reproducibility

Set a random seed at the top of your config to ensure reproducible results:

```yaml
seed_everything: 42
```

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Training stops after 1 epoch with no error | NaN losses | Check data for NaN values; try `precision: 32`; lower the LR |
| `Could not instantiate model` error | Wrong model name or download issue | Check the model name with `BACKBONE_REGISTRY`; check `HF_HOME` disk space |
| Dataset length is 0 | Wrong paths or file patterns | Verify `train_data_root`, `img_grep`, and `label_grep` |
| CUDA out of memory | Batch/chip size too large | Reduce `batch_size`, chip size, or use `precision: 16-mixed` |
| Size mismatch during inference | Padding issue | Add `tiled_inference_parameters` to your config |
| Poor performance on minority classes | Class imbalance | Switch to Dice loss; monitor per-class metrics |

For more help, check the [FAQs](faqs.md) or open an issue on [GitHub](https://github.com/IBM/terratorch).

---

## Complete Working Examples

Ready-to-run examples with configuration files and notebooks are available in the `examples/` directory:

| Example | Task | Dataset | Link |
|---------|------|---------|------|
| Burn scar detection | Segmentation | HLS burn scars | [Tutorial](../tutorials/burn_scars_finetuning.md) |
| Flood mapping | Segmentation | SEN1Floods11 | `examples/segmentation/` |
| Land use classification | Classification | EuroSAT | `examples/classification/` |
| Biomass estimation | Pixel-wise regression | AGB dataset | `examples/pixelwise_regression/` |
| Elephant detection | Object detection | AED dataset | `examples/object_detection/` |
| Crop type mapping | Segmentation (multi-temporal) | PASTIS | `examples/multitemporal_data/` |
