"""
Unit tests for model components.

Note: These tests mock heavy weight loading to allow testing without
downloading large model files.
"""


import pytest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="Requires torch")


class TestPopulateArgs:
    """Tests for populate_args function."""

    def test_populate_args_defaults(self):
        """Test populate_args returns correct default values."""
        from rfdetr.main import populate_args

        args = populate_args()

        assert args.num_classes == 2
        assert args.batch_size == 2
        assert args.lr == 1e-4
        assert args.epochs == 12
        assert args.hidden_dim == 256
        assert args.num_queries == 300
        assert args.resolution == 640
        assert args.device == "cuda"

    def test_populate_args_custom_values(self):
        """Test populate_args accepts custom values."""
        from rfdetr.main import populate_args

        args = populate_args(
            num_classes=10,
            batch_size=8,
            epochs=50,
            resolution=560,
            device="cpu",
        )

        assert args.num_classes == 10
        assert args.batch_size == 8
        assert args.epochs == 50
        assert args.resolution == 560
        assert args.device == "cpu"

    def test_populate_args_mask_weights(self):
        """Test populate_args includes mask-specific weights."""
        from rfdetr.main import populate_args

        args = populate_args(
            spatial_backbone_weights="/path/to/spatial.pt",
            mask_weights="/path/to/mask.pt",
        )

        assert args.spatial_backbone_weights == "/path/to/spatial.pt"
        assert args.mask_weights == "/path/to/mask.pt"


class TestBoxOperations:
    """Tests for box utility functions."""

    def test_box_cxcywh_to_xyxy(self):
        """Test conversion from center format to corner format."""
        from rfdetr.util.box_ops import box_cxcywh_to_xyxy

        # Box: center_x=0.5, center_y=0.5, width=0.4, height=0.4
        boxes = torch.tensor([[0.5, 0.5, 0.4, 0.4]])

        xyxy = box_cxcywh_to_xyxy(boxes)

        # Expected: x1=0.3, y1=0.3, x2=0.7, y2=0.7
        expected = torch.tensor([[0.3, 0.3, 0.7, 0.7]])

        assert torch.allclose(xyxy, expected, atol=1e-6)

    def test_box_xyxy_to_cxcywh(self):
        """Test conversion from corner format to center format."""
        from rfdetr.util.box_ops import box_xyxy_to_cxcywh

        boxes = torch.tensor([[0.3, 0.3, 0.7, 0.7]])

        cxcywh = box_xyxy_to_cxcywh(boxes)

        expected = torch.tensor([[0.5, 0.5, 0.4, 0.4]])

        assert torch.allclose(cxcywh, expected, atol=1e-6)

    def test_box_area(self):
        """Test box area calculation."""
        from rfdetr.util.box_ops import box_area

        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])

        area = box_area(boxes)

        assert area.item() == 100.0

    def test_generalized_box_iou(self):
        """Test generalized IoU calculation."""
        from rfdetr.util.box_ops import generalized_box_iou

        boxes1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        boxes2 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])

        giou = generalized_box_iou(boxes1, boxes2)

        # Same boxes should have IoU of 1.0
        assert torch.allclose(giou, torch.tensor([[1.0]]), atol=1e-6)


class TestMLP:
    """Tests for MLP class."""

    def test_mlp_forward(self):
        """Test MLP forward pass."""
        from rfdetr.models.lwdetr import MLP

        mlp = MLP(input_dim=256, hidden_dim=512, output_dim=4, num_layers=3)

        x = torch.randn(2, 100, 256)
        output = mlp(x)

        assert output.shape == (2, 100, 4)

    def test_mlp_single_layer(self):
        """Test MLP with single layer."""
        from rfdetr.models.lwdetr import MLP

        mlp = MLP(input_dim=256, hidden_dim=256, output_dim=128, num_layers=1)

        x = torch.randn(1, 50, 256)
        output = mlp(x)

        assert output.shape == (1, 50, 128)


class TestPositionEncoding:
    """Tests for position encoding."""

    def test_sine_position_encoding_shape(self):
        """Test sine position encoding output shape."""
        from rfdetr.models.position_encoding import PositionEmbeddingSine

        pos_embed = PositionEmbeddingSine(num_pos_feats=128, normalize=True)

        # Create dummy nested tensor
        from rfdetr.util.misc import NestedTensor

        tensor = torch.randn(2, 256, 40, 40)
        mask = torch.zeros(2, 40, 40, dtype=torch.bool)
        nested = NestedTensor(tensor, mask)

        output = pos_embed(nested)

        assert output.shape == (2, 256, 40, 40)


class TestSetCriterion:
    """Tests for loss functions."""

    def test_sigmoid_focal_loss(self):
        """Test sigmoid focal loss computation."""
        from rfdetr.models.lwdetr import sigmoid_focal_loss

        inputs = torch.randn(2, 100, 10)
        targets = torch.zeros(2, 100, 10)
        targets[:, :5, :] = 1  # Some positive examples

        loss = sigmoid_focal_loss(inputs, targets, num_boxes=10)

        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0  # Loss should be non-negative

    def test_dice_loss(self):
        """Test dice loss computation."""
        from rfdetr.models.lwdetr import dice_loss

        inputs = torch.randn(10, 100)  # 10 masks, 100 pixels each
        targets = torch.rand(10, 100) > 0.5  # Binary masks
        targets = targets.float()

        loss = dice_loss(inputs, targets, num_boxes=10)

        assert loss.dim() == 0  # Scalar
        assert 0 <= loss.item() <= 2  # Dice loss is in [0, 2]


class TestPostProcess:
    """Tests for post-processing."""

    def test_postprocess_output_format(self):
        """Test PostProcess output format."""
        from rfdetr.models.lwdetr import PostProcess

        postprocessor = PostProcess(num_select=100)

        # Create dummy outputs
        outputs = {
            "pred_logits": torch.randn(2, 300, 91),
            "pred_boxes": torch.rand(2, 300, 4),
        }
        target_sizes = torch.tensor([[480, 640], [480, 640]])

        results = postprocessor(outputs, target_sizes)

        assert len(results) == 2  # Batch size 2
        assert "scores" in results[0]
        assert "labels" in results[0]
        assert "boxes" in results[0]
        assert results[0]["boxes"].shape[-1] == 4  # xyxy format


class TestIntegration:
    """Integration tests (require mocking)."""

    @pytest.mark.skip(reason="Requires actual weight files")
    def test_model_forward_pass(self):
        """Test full model forward pass."""
        # This test would require actual weight files
        # Skip by default, can be enabled for integration testing
        pass
