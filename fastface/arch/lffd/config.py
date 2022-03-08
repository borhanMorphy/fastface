from typing import Tuple
from ...config import ArchConfig


class LFFDConfig(ArchConfig):
    """LFFD architecture configuration
    default configuration is `original`
    """

    # main configs
    input_width: int = 640
    input_height: int = 640
    input_channel: int = 3
    backbone: str = "lffd-v1"

    # lffd specific configs
    head_in_features: Tuple = (64, 64, 64, 64, 128, 128, 128, 128)
    head_out_features: Tuple = (128, 128, 128, 128, 128, 128, 128, 128)
    rf_sizes: Tuple = (15, 20, 40, 70, 110, 250, 400, 560)
    rf_strides: Tuple = (4, 4, 8, 8, 16, 32, 32, 32)
    min_face_size: int = 10
