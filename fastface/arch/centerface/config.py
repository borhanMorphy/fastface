from typing import List, Union

from ...config import ArchConfig


class CenterFaceConfig(ArchConfig):
    """CenterFace architecture configuration
    default configuration is original
    """

    # name of the architecture
    arch: str = "centerface"
    # name of the configuration
    name: str = "original"

    # preprocess
    mean: Union[float, List] = 0.0
    std: Union[float, List] = 1.0
    normalized_input: bool = True

    # main configs
    input_width: int = 512
    input_height: int = 512
    input_channel: int = 3
    backbone: str = "mobilenetv2"

    fpn_in_features: List[int] = [24, 32, 96, 320]
    fpn_out_feature: int = 24
    fpn_upsample_method: str = "deconv"
    output_stride: int = 4
    # change this if dataset contains different number of landmark annotations
    num_landmarks: int = 5

    # training hparams
    batch_size: int = 8
    learning_rate: float = 0.0005
    scheduler_lr_milestones: List[int] = [90, 120]
    scheduler_lr_gamma: float = 0.1

    loss_lambda_cls: float = 1.0
    loss_lambda_offset: float = 1.0
    loss_lambda_wh: float = 0.1
    loss_lambda_landmark: float = 0.1

    cls_loss_beta: float = 4.0
    cls_loss_alpha: float = 0.5
    cls_loss_gamma: float = 2.0

    max_epochs: int = 140
    img_size: int = 800
    min_face_area: int = 64
