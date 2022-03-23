from typing import List, Tuple, Union

from ...config import ArchConfig


class LFFDConfig(ArchConfig):
    """LFFD architecture configuration
    default configuration is `original`
    """

    # name of the architecture in the factory
    arch: str = "lffd"
    # name of the configuration in the factory
    name: str = "original"

    # preprocess
    mean: Union[float, List] = 127.5
    std: Union[float, List] = 127.5
    normalized_input: bool = False

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

    # training hparams
    batch_size: int = 32
    max_epochs: int = 800
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 0.00001
    scheduler_milestones: List[int] = [200, 400, 600]
    scheduler_gamma: float = 0.1
    hard_neg_mining_ratio: int = 5
