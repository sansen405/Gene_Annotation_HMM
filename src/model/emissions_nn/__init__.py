"""Neural emission models for structured biological sequence modeling.

Pluggable DilatedCNN / BiLSTM / Transformer backbones, multi-task heads,
calibration utilities, masked LM pretraining, segment CRF, and CORAL adaptation.
"""

from .backbones import BACKBONE_NAMES, build_encoder, one_hot_encode_windows
from .calibration import expected_calibration_error, mc_dropout_predict
from .coral import coral_loss
from .models import MultiTaskEmissionModel, SpliceEmissionModel, StartEmissionModel
from .segment_crf import SegmentCRF
from .score_diagnostics import write_mc_dropout_diagnostics

__all__ = [
    "BACKBONE_NAMES",
    "build_encoder",
    "one_hot_encode_windows",
    "expected_calibration_error",
    "mc_dropout_predict",
    "coral_loss",
    "MultiTaskEmissionModel",
    "SpliceEmissionModel",
    "StartEmissionModel",
    "SegmentCRF",
    "write_mc_dropout_diagnostics",
]
