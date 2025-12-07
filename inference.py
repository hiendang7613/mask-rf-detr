#!/usr/bin/env python3
"""
Mask-RF-DETR Inference Script

This script performs instance segmentation inference using a trained Mask-RF-DETR model.

Usage:
    python inference.py --checkpoint path/to/checkpoint.pth --image path/to/image.jpg
    python inference.py --checkpoint path/to/checkpoint.pth --input_dir path/to/images/ --output_dir results/

Environment Variables:
    MASK_RFDETR_SPATIAL_BACKBONE: Path to SAM2 Hiera backbone weights
    MASK_RFDETR_MASK_WEIGHTS: Path to mask weights (optional)
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image

try:
    import supervision as sv
except ImportError:
    print("Please install supervision: pip install supervision")
    sys.exit(1)


def get_device() -> str:
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(
    checkpoint_path: str,
    device: str = None,
    spatial_backbone_weights: str = None,
) -> Tuple[torch.nn.Module, dict]:
    """
    Load a trained Mask-RF-DETR model from checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        device: Device to load the model on (cuda/mps/cpu)
        spatial_backbone_weights: Path to SAM2 Hiera backbone weights
        
    Returns:
        Tuple of (model, config_dict)
    """
    if device is None:
        device = get_device()
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract args from checkpoint if available
    args = checkpoint.get('args', None)
    
    # Get model configuration
    from rfdetr.main import populate_args
    from rfdetr.models import build_model
    
    config_kwargs = {}
    if args is not None:
        # Use saved args
        config_kwargs = {
            'encoder': getattr(args, 'encoder', 'dinov2_windowed_small'),
            'hidden_dim': getattr(args, 'hidden_dim', 256),
            'num_classes': getattr(args, 'num_classes', 90),
            'num_queries': getattr(args, 'num_queries', 300),
            'dec_layers': getattr(args, 'dec_layers', 5),
            'two_stage': getattr(args, 'two_stage', True),
            'projector_scale': getattr(args, 'projector_scale', ['P4']),
            'out_feature_indexes': getattr(args, 'out_feature_indexes', [2, 5, 8, 11]),
            'resolution': getattr(args, 'resolution', 560),
            'sa_nheads': getattr(args, 'sa_nheads', 8),
            'ca_nheads': getattr(args, 'ca_nheads', 16),
            'dec_n_points': getattr(args, 'dec_n_points', 2),
            'bbox_reparam': getattr(args, 'bbox_reparam', True),
            'lite_refpoint_refine': getattr(args, 'lite_refpoint_refine', True),
            'group_detr': getattr(args, 'group_detr', 9),
        }
    
    config_kwargs['device'] = device
    config_kwargs['pretrain_weights'] = None  # We'll load weights manually
    if spatial_backbone_weights:
        config_kwargs['spatial_backbone_weights'] = spatial_backbone_weights
    
    # Build model
    model_args = populate_args(**config_kwargs)
    model = build_model(model_args)
    
    # Load weights
    state_dict = checkpoint.get('model', checkpoint)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    
    model = model.to(device)
    model.eval()
    
    # Get class names if available
    class_names = None
    if args is not None and hasattr(args, 'class_names'):
        class_names = args.class_names
    
    config = {
        'device': device,
        'resolution': config_kwargs.get('resolution', 560),
        'class_names': class_names,
        'num_classes': config_kwargs.get('num_classes', 90),
    }
    
    return model, config


def preprocess_image(
    image: Union[str, Image.Image, np.ndarray],
    resolution: int = 560,
    device: str = "cpu",
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Preprocess an image for inference.
    
    Args:
        image: Input image (path, PIL Image, or numpy array)
        resolution: Target resolution
        device: Device to move tensor to
        
    Returns:
        Tuple of (preprocessed tensor, original size (h, w))
    """
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    orig_size = (image.height, image.width)
    
    # Convert to tensor
    img_tensor = F.to_tensor(image)
    
    # Normalize
    img_tensor = F.normalize(img_tensor, means, stds)
    
    # Resize
    img_tensor = F.resize(img_tensor, (resolution, resolution))
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    return img_tensor, orig_size


def postprocess_outputs(
    outputs: dict,
    orig_size: Tuple[int, int],
    threshold: float = 0.5,
    num_select: int = 300,
) -> sv.Detections:
    """
    Postprocess model outputs to get detections.
    
    Args:
        outputs: Model output dictionary
        orig_size: Original image size (h, w)
        threshold: Confidence threshold
        num_select: Number of top predictions to select
        
    Returns:
        sv.Detections object with boxes, masks, scores, and class_ids
    """
    pred_logits = outputs['pred_logits']
    pred_boxes = outputs['pred_boxes']
    pred_masks = outputs.get('pred_masks', None)
    
    # Get probabilities
    prob = pred_logits.sigmoid()
    
    # Select top predictions
    topk_values, topk_indexes = torch.topk(
        prob.view(pred_logits.shape[0], -1), 
        min(num_select, prob.shape[1] * prob.shape[2]), 
        dim=1
    )
    
    scores = topk_values[0]
    topk_boxes = topk_indexes[0] // pred_logits.shape[2]
    labels = topk_indexes[0] % pred_logits.shape[2]
    
    # Convert boxes from cxcywh to xyxy
    boxes = pred_boxes[0, topk_boxes]
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes = torch.stack([x1, y1, x2, y2], dim=-1)
    
    # Scale to original image size
    h_orig, w_orig = orig_size
    scale = torch.tensor([w_orig, h_orig, w_orig, h_orig], device=boxes.device)
    boxes = boxes * scale
    
    # Filter by threshold
    keep = scores > threshold
    scores = scores[keep]
    labels = labels[keep]
    boxes = boxes[keep]
    
    # Process masks if available
    masks = None
    if pred_masks is not None:
        selected_masks = pred_masks[0, topk_boxes[keep]]
        # Resize masks to original image size
        masks = torch.nn.functional.interpolate(
            selected_masks.unsqueeze(1).float(),
            size=(h_orig, w_orig),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)
        masks = (masks.sigmoid() > 0.5).cpu().numpy()
    
    # Create detections
    detections = sv.Detections(
        xyxy=boxes.cpu().numpy(),
        confidence=scores.cpu().numpy(),
        class_id=labels.cpu().numpy(),
        mask=masks,
    )
    
    return detections


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    image: Union[str, Image.Image, np.ndarray],
    config: dict,
    threshold: float = 0.5,
) -> sv.Detections:
    """
    Run inference on a single image.
    
    Args:
        model: The loaded model
        image: Input image
        config: Model configuration
        threshold: Confidence threshold
        
    Returns:
        sv.Detections object
    """
    device = config['device']
    resolution = config['resolution']
    
    # Preprocess
    img_tensor, orig_size = preprocess_image(image, resolution, device)
    
    # Run model
    outputs = model(img_tensor)
    
    # Postprocess
    detections = postprocess_outputs(outputs, orig_size, threshold)
    
    return detections


def visualize_detections(
    image: Union[str, Image.Image, np.ndarray],
    detections: sv.Detections,
    class_names: Optional[List[str]] = None,
    output_path: Optional[str] = None,
) -> np.ndarray:
    """
    Visualize detections on an image.
    
    Args:
        image: Input image
        detections: Detections to visualize
        class_names: Optional list of class names
        output_path: Optional path to save the visualization
        
    Returns:
        Annotated image as numpy array
    """
    if isinstance(image, str):
        image = np.array(Image.open(image).convert('RGB'))
    elif isinstance(image, Image.Image):
        image = np.array(image.convert('RGB'))
    
    # Create annotators
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()
    label_annotator = sv.LabelAnnotator()
    
    # Create labels
    if class_names:
        labels = [
            f"{class_names[class_id]} {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]
    else:
        labels = [
            f"class_{class_id} {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]
    
    # Annotate
    annotated = image.copy()
    if detections.mask is not None:
        annotated = mask_annotator.annotate(annotated, detections)
    annotated = box_annotator.annotate(annotated, detections)
    annotated = label_annotator.annotate(annotated, detections, labels=labels)
    
    if output_path:
        Image.fromarray(annotated).save(output_path)
        print(f"Saved visualization to: {output_path}")
    
    return annotated


def main():
    parser = argparse.ArgumentParser(
        description="Mask-RF-DETR Inference Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        help="Path to a single image for inference"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Path to directory containing images"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="inference_output",
        help="Directory to save results"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.5,
        help="Confidence threshold for detections"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on (cuda/mps/cpu)"
    )
    parser.add_argument(
        "--spatial_backbone_weights",
        type=str,
        default=None,
        help="Path to SAM2 Hiera backbone weights"
    )
    parser.add_argument(
        "--no_visualize",
        action="store_true",
        help="Skip visualization, only save detections as JSON"
    )
    
    args = parser.parse_args()
    
    if not args.image and not args.input_dir:
        parser.error("Either --image or --input_dir must be provided")
    
    # Load model
    model, config = load_model(
        args.checkpoint,
        device=args.device,
        spatial_backbone_weights=args.spatial_backbone_weights,
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get image paths
    image_paths = []
    if args.image:
        image_paths.append(Path(args.image))
    if args.input_dir:
        input_dir = Path(args.input_dir)
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']:
            image_paths.extend(input_dir.glob(ext))
            image_paths.extend(input_dir.glob(ext.upper()))
    
    print(f"Found {len(image_paths)} images to process")
    
    # Process images
    results = []
    for img_path in image_paths:
        print(f"Processing: {img_path}")
        
        # Run inference
        detections = run_inference(model, str(img_path), config, args.threshold)
        
        print(f"  Found {len(detections)} detections")
        
        # Save results
        result = {
            'image': str(img_path),
            'num_detections': len(detections),
            'detections': []
        }
        
        for i in range(len(detections)):
            det = {
                'class_id': int(detections.class_id[i]),
                'confidence': float(detections.confidence[i]),
                'bbox': detections.xyxy[i].tolist(),
            }
            if config['class_names'] and detections.class_id[i] < len(config['class_names']):
                det['class_name'] = config['class_names'][detections.class_id[i]]
            result['detections'].append(det)
        
        results.append(result)
        
        # Visualize
        if not args.no_visualize:
            output_path = output_dir / f"{img_path.stem}_annotated{img_path.suffix}"
            visualize_detections(
                str(img_path),
                detections,
                class_names=config['class_names'],
                output_path=str(output_path),
            )
    
    # Save results JSON
    import json
    results_path = output_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to: {results_path}")


if __name__ == "__main__":
    main()

