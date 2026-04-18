# transforms_v2.py – Conditional DETR / RF‑DETR data‑augmentation utils (v2)
# ------------------------------------------------------------------------
# Copyright (c) 2025
# ------------------------------------------------------------------------

from __future__ import annotations

import copy
import random
from typing import Iterable, List, Tuple, Union

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
from torch import Tensor

__all__ = [
    # helpers
    "interpolate",
    "box_xyxy_to_cxcywh",
    "box_cxcywh_to_xyxy",
    "crop",
    "hflip",
    "resize",
    "pad",
    # transform classes
    "RandomRotation90",
    "RandomCrop",
    "RandomSizeCrop",
    "CenterCrop",
    "RandomHorizontalFlip",
    "RandomResize",
    "RandomPad",
    "RandomSelect",
    "ToTensor",
    "RandomErasing",
    "Normalize",
    "Compose",
    "SquareResize",
]

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------


def _clone_target(tgt: dict | None) -> dict | None:
    """Deep‑ish clone: clone tất cả Tensor, giữ nguyên kiểu dữ liệu khác."""
    if tgt is None:
        return None
    return {
        k: v.clone() if torch.is_tensor(v) else copy.deepcopy(v) for k, v in tgt.items()
    }


def interpolate(
    x: Tensor,
    size: Tuple[int, int] | None = None,
    scale_factor: float | None = None,
    mode: str = "nearest",
    align_corners: bool | None = None,
) -> Tensor:
    """Wrapper quanh ``torch.nn.functional.interpolate`` để xử lý batch rỗng."""
    if x.shape[0] == 0:
        return x
    return torch.nn.functional.interpolate(
        x, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners
    )


# -----------------------------------------------------------------------------
# Box conversions
# -----------------------------------------------------------------------------


def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack((cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h), dim=-1)


def box_xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    x0, y0, x1, y1 = boxes.unbind(-1)
    return torch.stack(((x0 + x1) * 0.5, (y0 + y1) * 0.5, x1 - x0, y1 - y0), dim=-1)


# -----------------------------------------------------------------------------
# Core geometric ops (functional API)
# -----------------------------------------------------------------------------


def crop(image: Image.Image, target: dict, region: Tuple[int, int, int, int]):
    """Crop PIL image & target; ``region = (top, left, height, width)``."""
    cropped = F.crop(image, *region)
    top, left, h, w = region

    tgt = _clone_target(target)
    tgt["size"] = torch.tensor([h, w])

    # --- boxes -------------------------------------------------
    if "boxes" in tgt and tgt["boxes"].numel():
        boxes = tgt["boxes"].to(torch.float32)
        # Shift
        boxes -= torch.tensor([left, top, left, top], device=boxes.device)
        # Clamp to crop
        boxes = boxes.view(-1, 2, 2).clamp(min=0)
        boxes = boxes.clamp(max=torch.tensor([w, h], device=boxes.device)).view(-1, 4)
        tgt["boxes"] = boxes
        tgt["area"] = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (
            boxes[:, 3] - boxes[:, 1]
        ).clamp(min=0)

    # --- masks -------------------------------------------------
    if "masks" in tgt and tgt["masks"].numel():
        tgt["masks"] = tgt["masks"][:, top : top + h, left : left + w]

    # --- filter zero‑area / empty masks ------------------------
    if "boxes" in tgt:
        keep = tgt["area"] > 1
        for k in list(tgt.keys()):
            if torch.is_tensor(tgt[k]) and tgt[k].shape[0] == keep.shape[0]:
                tgt[k] = tgt[k][keep]

    return cropped, tgt


def hflip(image: Image.Image, target: dict):
    flipped = F.hflip(image)
    w, _ = image.size
    tgt = _clone_target(target)

    if "boxes" in tgt and tgt["boxes"].numel():
        boxes = tgt["boxes"].to(torch.float32)
        boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        tgt["boxes"] = boxes

    if "masks" in tgt and tgt["masks"].numel():
        tgt["masks"] = tgt["masks"].flip(-1)

    return flipped, tgt


def _get_resize_shape(
    img_size: Tuple[int, int],
    size: Union[int, Tuple[int, int]],
    max_size: int | None,
) -> Tuple[int, int]:
    """Return (new_h, new_w) giữ tỉ lệ nếu ``size`` là int."""
    w, h = img_size  # PIL trả (w, h)
    if isinstance(size, (tuple, list)):
        return size  # (h, w)

    min_orig, max_orig = float(min(w, h)), float(max(w, h))
    if max_size and (max_orig / min_orig) * size > max_size:
        size = int(round(max_size * min_orig / max_orig))

    if (w <= h and w == size) or (h <= w and h == size):
        return h, w

    if w < h:
        return int(size * h / w), size
    return size, int(size * w / h)


def resize(
    image: Image.Image,
    target: dict | None,
    size: Union[int, Tuple[int, int]],
    max_size: int | None = None,
):
    new_h, new_w = _get_resize_shape(image.size, size, max_size)
    rescaled = F.resize(image, (new_h, new_w))

    if target is None:
        return rescaled, None

    tgt = _clone_target(target)
    ratio_w, ratio_h = new_w / image.width, new_h / image.height

    if "boxes" in tgt and tgt["boxes"].numel():
        tgt["boxes"] = tgt["boxes"].to(torch.float32) * torch.tensor(
            [ratio_w, ratio_h, ratio_w, ratio_h], device=tgt["boxes"].device
        )
        tgt["area"] = tgt.get("area", torch.zeros(0, device=tgt["boxes"].device)) * (
            ratio_w * ratio_h
        )

    tgt["size"] = torch.tensor([new_h, new_w])

    if "masks" in tgt and tgt["masks"].numel():
        tgt["masks"] = interpolate(
            tgt["masks"].unsqueeze(1).float(), size=(new_h, new_w), mode="nearest"
        )[:, 0].to(dtype=torch.bool)

    return rescaled, tgt


def pad(image: Image.Image | Tensor, target: dict | None, padding: Tuple[int, int]):
    """Pad bottom‑right (pad_x, pad_y)."""
    padded = F.pad(image, (0, 0, padding[0], padding[1]))

    if target is None:
        return padded, None

    tgt = _clone_target(target)
    if isinstance(padded, Image.Image):
        w, h = padded.size
    else:  # Tensor C×H×W
        h, w = padded.shape[-2:]

    tgt["size"] = torch.tensor([h, w])

    if "masks" in tgt and tgt["masks"].numel():
        tgt["masks"] = torch.nn.functional.pad(
            tgt["masks"], (0, padding[0], 0, padding[1])
        )
    return padded, tgt


# -----------------------------------------------------------------------------
# Transform class objects
# -----------------------------------------------------------------------------


class RandomRotation90:
    """Rotate 0/90/180/270° ngẫu nhiên."""

    def __call__(self, img: Image.Image, tgt: dict):
        k = random.randint(0, 3)
        if k == 0:
            return img, tgt

        angle = k * 90
        img = F.rotate(img, angle, expand=True, fill=255)
        h, w = img.height, img.width

        out = _clone_target(tgt)

        if "boxes" in out and out["boxes"].numel():
            boxes = out["boxes"].to(torch.float32)
            x0, y0, x1, y1 = boxes.unbind(-1)
            if k == 1:  # 90°
                out["boxes"] = torch.stack([y0, w - x1, y1, w - x0], -1)
            elif k == 2:  # 180°
                out["boxes"] = torch.stack([w - x1, h - y1, w - x0, h - y0], -1)
            else:  # 270°
                out["boxes"] = torch.stack([h - y1, x0, h - y0, x1], -1)
        if "masks" in out and out["masks"].numel():
            out["masks"] = torch.rot90(out["masks"], k, (-2, -1))

        out["size"] = torch.tensor([h, w])
        return img, out


class RandomCrop:
    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, img: Image.Image, tgt: dict):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, tgt, region)


class RandomSizeCrop:
    def __init__(self, min_size: int, max_size: int):
        self.min = min_size
        self.max = max_size

    def __call__(self, img: Image.Image, tgt: dict):
        w = random.randint(self.min, min(img.width, self.max))
        h = random.randint(self.min, min(img.height, self.max))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, tgt, region)


class CenterCrop:
    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, img: Image.Image, tgt: dict):
        ih, iw = img.height, img.width
        ch, cw = self.size
        top = int(round((ih - ch) * 0.5))
        left = int(round((iw - cw) * 0.5))
        return crop(img, tgt, (top, left, ch, cw))


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img, tgt):
        return hflip(img, tgt) if random.random() < self.p else (img, tgt)


class RandomResize:
    def __init__(
        self, sizes: Iterable[int | Tuple[int, int]], max_size: int | None = None
    ):
        self.sizes = list(sizes)
        self.max_size = max_size

    def __call__(self, img, tgt):
        return resize(img, tgt, random.choice(self.sizes), self.max_size)


class RandomPad:
    def __init__(self, max_pad: int):
        self.max_pad = max_pad

    def __call__(self, img, tgt):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, tgt, (pad_x, pad_y))


class RandomSelect:
    def __init__(self, t1, t2, p: float = 0.5):
        self.t1, self.t2, self.p = t1, t2, p

    def __call__(self, img, tgt):
        return self.t1(img, tgt) if random.random() < self.p else self.t2(img, tgt)


class ToTensor:
    def __call__(self, img, tgt):
        return F.to_tensor(img), tgt


class RandomErasing:
    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, tgt):
        return self.eraser(img), tgt


class Normalize:
    def __init__(self, mean: List[float], std: List[float], reparam: bool = False):
        self.mean, self.std, self.reparam = mean, std, reparam

    def __call__(self, img: Tensor, tgt: dict | None = None):
        img = F.normalize(img, self.mean, self.std)
        if tgt is None:
            return img, None

        out = _clone_target(tgt)
        if "boxes" in out and out["boxes"].numel():
            boxes = box_xyxy_to_cxcywh(out["boxes"].to(torch.float32))
            if not self.reparam:
                h, w = img.shape[-2:]
                boxes /= torch.tensor([w, h, w, h], device=boxes.device)
            out["boxes"] = boxes
        return img, out


class Compose:
    def __init__(self, transforms: Iterable):
        self.tfms = list(transforms)

    def __call__(self, img, tgt):
        for t in self.tfms:
            img, tgt = t(img, tgt)
        return img, tgt

    def __repr__(self):
        body = ",\n    ".join(repr(t) for t in self.tfms)
        return f"{self.__class__.__name__}(\n    {body}\n)"


class SquareResize:
    def __init__(self, sizes: Iterable[int]):
        self.sizes = list(sizes)

    def __call__(self, img: Image.Image, tgt: dict | None = None):
        s = random.choice(self.sizes)
        orig_w, orig_h = img.size
        img = img.resize((s, s), Image.BILINEAR)

        if tgt is None:
            return img, None

        out = _clone_target(tgt)
        scale = (
            torch.tensor(
                [s / orig_w, s / orig_h, s / orig_w, s / orig_h],
                device=out["boxes"].device,
            )
            if "boxes" in out
            else None
        )

        if scale is not None and out["boxes"].numel():
            out["boxes"] = out["boxes"] * scale
        if "masks" in out and out["masks"].numel():
            out["masks"] = interpolate(
                out["masks"].unsqueeze(0).float(), size=(s, s), mode="nearest"
            )[0].to(dtype=torch.bool)
        out["size"] = torch.tensor([s, s])
        return img, out
