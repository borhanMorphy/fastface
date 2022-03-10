from .module import LFFD
from .config import LFFDConfig

from ...factory import _Factory

_Factory.register(
    name="lffd",
    cls=LFFD,
    configs=dict(
        # default configuration
        original=LFFDConfig(),

        slim=LFFDConfig(
            backbone="lffd-v2",
            head_in_features=(64, 64, 64, 128, 128),
            head_out_features=(128, 128, 128, 128, 128),
            rf_sizes=(20, 40, 80, 160, 320),
            rf_strides=(4, 8, 16, 32, 64),
            batch_size=16,
        )
    )
)
