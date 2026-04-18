"""
Pytest configuration and fixtures for Mask-RF-DETR tests.

Note: These tests require PyTorch and other dependencies to be installed.
Run: pip install torch pytest pydantic
"""


import pytest

# Check if torch is available
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "requires_torch: mark test as requiring PyTorch")


def pytest_collection_modifyitems(config, items):
    """Skip tests that require torch if it's not available."""
    if HAS_TORCH:
        return

    skip_torch = pytest.mark.skip(reason="PyTorch not installed")
    for item in items:
        if "requires_torch" in item.keywords:
            item.add_marker(skip_torch)


@pytest.fixture
def device():
    """Return available device for testing."""
    if not HAS_TORCH:
        pytest.skip("PyTorch not installed")
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@pytest.fixture
def dummy_image_tensor():
    """Create a dummy image tensor for testing."""
    if not HAS_TORCH:
        pytest.skip("PyTorch not installed")
    return torch.randn(1, 3, 560, 560)


@pytest.fixture
def dummy_nested_tensor():
    """Create a dummy NestedTensor for testing."""
    if not HAS_TORCH:
        pytest.skip("PyTorch not installed")
    from rfdetr.util.misc import NestedTensor

    tensor = torch.randn(2, 3, 560, 560)
    mask = torch.zeros(2, 560, 560, dtype=torch.bool)

    return NestedTensor(tensor, mask)


@pytest.fixture
def sample_boxes():
    """Create sample bounding boxes in various formats."""
    if not HAS_TORCH:
        pytest.skip("PyTorch not installed")
    # cxcywh format (center_x, center_y, width, height) - normalized
    cxcywh = torch.tensor(
        [
            [0.5, 0.5, 0.4, 0.4],
            [0.3, 0.7, 0.2, 0.3],
        ]
    )

    # xyxy format (x1, y1, x2, y2) - normalized
    xyxy = torch.tensor(
        [
            [0.3, 0.3, 0.7, 0.7],
            [0.2, 0.55, 0.4, 0.85],
        ]
    )

    return {"cxcywh": cxcywh, "xyxy": xyxy}


@pytest.fixture
def sample_targets():
    """Create sample targets for loss computation."""
    if not HAS_TORCH:
        pytest.skip("PyTorch not installed")
    return [
        {
            "labels": torch.tensor([1, 2, 3]),
            "boxes": torch.tensor(
                [
                    [0.5, 0.5, 0.2, 0.2],
                    [0.3, 0.3, 0.1, 0.1],
                    [0.7, 0.7, 0.3, 0.3],
                ]
            ),
            "masks": torch.rand(3, 560, 560) > 0.5,
        },
        {
            "labels": torch.tensor([1, 4]),
            "boxes": torch.tensor(
                [
                    [0.4, 0.4, 0.2, 0.2],
                    [0.6, 0.6, 0.15, 0.15],
                ]
            ),
            "masks": torch.rand(2, 560, 560) > 0.5,
        },
    ]
