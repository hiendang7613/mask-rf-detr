"""
Unit tests for configuration classes.
"""

import pytest

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from pydantic import ValidationError

    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    ValidationError = None

pytestmark = pytest.mark.skipif(
    not HAS_TORCH or not HAS_PYDANTIC, reason="Requires torch and pydantic"
)


class TestModelConfig:
    """Tests for ModelConfig and derived classes."""

    def test_model_config_defaults(self):
        """Test that ModelConfig has correct default values."""
        from rfdetr.config import ModelConfig

        # ModelConfig requires encoder, out_feature_indexes, projector_scale,
        # hidden_dim, sa_nheads, ca_nheads, dec_n_points
        config = ModelConfig(
            encoder="dinov2_windowed_small",
            out_feature_indexes=[2, 5, 8, 11],
            projector_scale=["P4"],
            hidden_dim=256,
            sa_nheads=8,
            ca_nheads=16,
            dec_n_points=2,
        )

        assert config.dec_layers == 5
        assert config.two_stage is True
        assert config.bbox_reparam is True
        assert config.lite_refpoint_refine is True
        assert config.layer_norm is True
        assert config.amp is True
        assert config.num_classes == 90
        assert config.resolution == 560
        assert config.group_detr == 9
        assert config.gradient_checkpointing is False

    def test_rfdetr_base_config_defaults(self):
        """Test RFDETRBaseConfig has correct defaults."""
        from rfdetr.config import RFDETRBaseConfig

        config = RFDETRBaseConfig()

        assert config.encoder == "dinov2_windowed_small"
        assert config.hidden_dim == 256
        assert config.sa_nheads == 8
        assert config.ca_nheads == 16
        assert config.dec_n_points == 2
        assert config.num_queries == 300
        assert config.num_select == 300
        assert config.projector_scale == ["P4"]
        assert config.out_feature_indexes == [2, 5, 8, 11]
        assert config.pretrain_weights == "rf-detr-base.pth"

    def test_rfdetr_large_config_defaults(self):
        """Test RFDETRLargeConfig has correct defaults."""
        from rfdetr.config import RFDETRLargeConfig

        config = RFDETRLargeConfig()

        assert config.encoder == "dinov2_windowed_base"
        assert config.hidden_dim == 384
        assert config.sa_nheads == 12
        assert config.ca_nheads == 24
        assert config.dec_n_points == 4
        assert config.projector_scale == ["P3", "P5"]
        assert config.pretrain_weights == "rf-detr-large.pth"

    def test_model_config_encoder_validation(self):
        """Test that encoder must be one of the allowed values."""
        from rfdetr.config import ModelConfig

        with pytest.raises(ValidationError):
            ModelConfig(
                encoder="invalid_encoder",
                out_feature_indexes=[2, 5, 8, 11],
                projector_scale=["P4"],
                hidden_dim=256,
                sa_nheads=8,
                ca_nheads=16,
                dec_n_points=2,
            )

    def test_model_config_device_validation(self):
        """Test that device must be one of cpu/cuda/mps."""
        from rfdetr.config import ModelConfig

        with pytest.raises(ValidationError):
            ModelConfig(
                encoder="dinov2_windowed_small",
                out_feature_indexes=[2, 5, 8, 11],
                projector_scale=["P4"],
                hidden_dim=256,
                sa_nheads=8,
                ca_nheads=16,
                dec_n_points=2,
                device="tpu",
            )

    def test_spatial_backbone_weights_config(self):
        """Test spatial_backbone_weights config field."""
        from rfdetr.config import ModelConfig

        config = ModelConfig(
            encoder="dinov2_windowed_small",
            out_feature_indexes=[2, 5, 8, 11],
            projector_scale=["P4"],
            hidden_dim=256,
            sa_nheads=8,
            ca_nheads=16,
            dec_n_points=2,
            spatial_backbone_weights="custom/path/weights.pt",
        )

        assert config.spatial_backbone_weights == "custom/path/weights.pt"


class TestTrainConfig:
    """Tests for TrainConfig class."""

    def test_train_config_defaults(self):
        """Test TrainConfig has correct defaults."""
        from rfdetr.config import TrainConfig

        config = TrainConfig(dataset_dir="/tmp/dataset")

        assert config.lr == 1e-4
        assert config.lr_encoder == 5e-5
        assert config.batch_size == 4
        assert config.grad_accum_steps == 4
        assert config.epochs == 100
        assert config.ema_decay == 0.993
        assert config.ema_tau == 100
        assert config.lr_drop == 100
        assert config.checkpoint_interval == 10
        assert config.warmup_epochs == 0
        assert config.lr_vit_layer_decay == 0.8
        assert config.lr_component_decay == 0.7
        assert config.drop_path == 0.0
        assert config.group_detr == 13
        assert config.ia_bce_loss is True
        assert config.cls_loss_coef == 1.0
        assert config.num_select == 300
        assert config.dataset_file == "roboflow"
        assert config.square_resize_div_64 is True
        assert config.output_dir == "output"
        assert config.multi_scale is True
        assert config.expanded_scales is True
        assert config.use_ema is True
        assert config.num_workers == 2
        assert config.weight_decay == 1e-4
        assert config.early_stopping is False
        assert config.early_stopping_patience == 10
        assert config.early_stopping_min_delta == 0.001
        assert config.early_stopping_use_ema is False
        assert config.tensorboard is True
        assert config.wandb is False
        assert config.run_test is True

    def test_train_config_requires_dataset_dir(self):
        """Test that dataset_dir is required."""
        from rfdetr.config import TrainConfig

        with pytest.raises(ValidationError):
            TrainConfig()  # Missing dataset_dir

    def test_train_config_dataset_file_validation(self):
        """Test dataset_file must be one of allowed values."""
        from rfdetr.config import TrainConfig

        with pytest.raises(ValidationError):
            TrainConfig(
                dataset_dir="/tmp/dataset",
                dataset_file="invalid_format",
            )
