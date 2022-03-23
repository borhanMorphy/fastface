from ...factory import _Factory
from .config import LFFDConfig
from .module import LFFD

_Factory.register(
    cls=LFFD,
    configs=[
        # default configuration
        LFFDConfig(),
        # another configuration defined in the paper
        LFFDConfig(
            name="slim",
            backbone="lffd-v2",
            head_in_features=(64, 64, 64, 128, 128),
            head_out_features=(128, 128, 128, 128, 128),
            rf_sizes=(20, 40, 80, 160, 320),
            rf_strides=(4, 8, 16, 32, 64),
            batch_size=16,
        ),
    ],
)
