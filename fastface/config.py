from typing import List, Union

from pydantic import BaseModel


class ArchConfig(BaseModel):
    """Generic base configuration for face detection architectures"""

    # name of the architecture
    arch: str
    # name of the configuration
    name: str

    # preprocess
    mean: Union[float, List] = 0.0
    std: Union[float, List] = 1.0
    normalized_input: bool = True

    # main configs
    input_width: int
    input_height: int
    input_channel: int
    backbone: str

    # training hparams
    batch_size: int
    learning_rate: float
    max_epochs: int
