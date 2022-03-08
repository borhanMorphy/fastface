from .module import LFFD
from .config import LFFDConfig

from ...factory import _Factory

_Factory.register(
    name="lffd",
    cls=LFFD,
    configs=dict(
        original=LFFDConfig(
            input_width=640,
            input_height=640,
            input_channel=3,
            backbone="lffd-v1",
            head_in_features=(64, 64, 64, 64, 128, 128, 128, 128),
            head_out_features=(128, 128, 128, 128, 128, 128, 128, 128),
            rf_sizes=(15, 20, 40, 70, 110, 250, 400, 560),
            rf_strides=(4, 4, 8, 8, 16, 32, 32, 32),
            min_face_size=10,
        ),

        slim=LFFDConfig(
            input_width=480,
            input_height=480,
            input_channel=3,
            backbone="lffd-v2",
            head_in_features=(64, 64, 64, 128, 128),
            head_out_features=(128, 128, 128, 128, 128),
            rf_sizes=(20, 40, 80, 160, 320),
            rf_strides=(4, 8, 16, 32, 64),
            min_face_size=10,
        )
    )
)
