# Mask-RF-DETR: Transformer-based Instance Segmentation

Mask-RF-DETR extends [RF-DETR](https://github.com/roboflow/rf-detr) with instance segmentation capabilities by integrating Mask2Former and SAM2 Hiera backbone.

![Architecture](https://github.com/user-attachments/assets/151c80bc-0ac1-4c0f-8712-e69f6029f070)

![Results](https://github.com/user-attachments/assets/450b857b-453e-486f-8948-2737378275e4)

## Features

- **Object Detection + Instance Segmentation**: Predicts both bounding boxes and pixel-level masks
- **DINOv2 Backbone**: Uses DINOv2 with windowed attention for efficient feature extraction
- **Mask2Former Decoder**: Leverages Mask2Former pixel decoder for high-quality masks
- **SAM2 Hiera Integration**: Uses SAM2's Hiera backbone for spatial features
- **Multiple Model Sizes**: Base and Large configurations available

## Installation

### Requirements

- Python >= 3.9
- PyTorch >= 1.13.0
- CUDA (recommended for training/inference)

### Install from source

```bash
git clone https://github.com/your-repo/Mask-rf-detr.git
cd Mask-rf-detr
pip install -e .
```

### Install dependencies only

```bash
pip install torch torchvision
pip install transformers supervision peft timm einops
pip install pycocotools scipy tqdm numpy pandas matplotlib
```

## Model Weights

### Required Weights

1. **SAM2 Hiera Backbone** (required for mask prediction):
   - Download from: [SAM2 Model Zoo](https://github.com/facebookresearch/segment-anything-2)
   - Expected file: `Hiera_sam2.1_hiera_base_plus.pt`
   - Set path via environment variable or config:
     ```bash
     export MASK_RFDETR_SPATIAL_BACKBONE=/path/to/Hiera_sam2.1_hiera_base_plus.pt
     ```

2. **RF-DETR Base Weights** (auto-downloaded):
   - `rf-detr-base.pth` - Base model pretrained on COCO
   - `rf-detr-large.pth` - Large model pretrained on COCO

### Optional Weights

- **Mask Weights**: Pre-trained mask head weights
  ```bash
  export MASK_RFDETR_MASK_WEIGHTS=/path/to/mask-weights.pt
  ```

## Quick Start

### Inference

```python
from rfdetr import RFDETRBase
import supervision as sv
from PIL import Image

# Initialize model
model = RFDETRBase(
    spatial_backbone_weights="weights/Hiera_sam2.1_hiera_base_plus.pt"
)

# Load trained checkpoint
checkpoint = torch.load("checkpoint_best_total.pth", map_location="cpu")
model.model.load_state_dict(checkpoint["model"], strict=False)

# Run inference
image = Image.open("test.jpg")
detections = model.predict(image, threshold=0.5)

# Visualize
annotator = sv.BoxAnnotator()
annotated = annotator.annotate(image, detections)
```

### Command-line Inference

```bash
python inference.py \
    --checkpoint path/to/checkpoint.pth \
    --image path/to/image.jpg \
    --output_dir results/ \
    --threshold 0.5
```

For batch processing:

```bash
python inference.py \
    --checkpoint path/to/checkpoint.pth \
    --input_dir path/to/images/ \
    --output_dir results/
```

### Training

```python
from rfdetr import RFDETRBase

model = RFDETRBase(
    spatial_backbone_weights="weights/Hiera_sam2.1_hiera_base_plus.pt"
)

model.train(
    dataset_dir="path/to/coco/dataset",
    epochs=100,
    batch_size=4,
    lr=1e-4,
    output_dir="output/",
)
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MASK_RFDETR_SPATIAL_BACKBONE` | Path to SAM2 Hiera weights | `weights/Hiera_sam2.1_hiera_base_plus.pt` |
| `MASK_RFDETR_MASK_WEIGHTS` | Path to mask weights (optional) | `weights/mask-rf-detr-coco.pt` |

### Model Configurations

**RFDETRBase**:
- Encoder: DINOv2 Small with windowed attention
- Hidden dim: 256
- Resolution: 560x560
- Queries: 300

**RFDETRLarge**:
- Encoder: DINOv2 Base with windowed attention
- Hidden dim: 384
- Resolution: 560x560
- Queries: 300

## Dataset Format

The model expects COCO format datasets:

```
dataset/
├── train/
│   ├── _annotations.coco.json
│   └── images...
├── valid/
│   ├── _annotations.coco.json
│   └── images...
└── test/
    ├── _annotations.coco.json
    └── images...
```

## Architecture Overview

```
Input Image
    │
    ├──► DINOv2 Backbone ──► Multi-scale Features
    │
    └──► SAM2 Hiera Backbone ──► Spatial Features
                                      │
                                      ▼
                              Mask2Former Pixel Decoder
                                      │
                                      ▼
                              LW-DETR Transformer Decoder
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
              Class Logits      Box Coords        Mask Logits
```

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## Acknowledgments

This project builds upon:
- [RF-DETR](https://github.com/roboflow/rf-detr) by Roboflow
- [LW-DETR](https://github.com/Atten4Vis/LW-DETR) by Baidu
- [Mask2Former](https://github.com/facebookresearch/Mask2Former) by Meta
- [SAM2](https://github.com/facebookresearch/segment-anything-2) by Meta
- [DINOv2](https://github.com/facebookresearch/dinov2) by Meta

## Citation

```bibtex
@software{mask_rfdetr,
  title = {Mask-RF-DETR: Transformer-based Instance Segmentation with RF-DETR},
  year = {2025},
  url = {https://github.com/your-repo/Mask-rf-detr}
}
```
