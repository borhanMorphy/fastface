from ...factory import _Factory
from .config import CenterFaceConfig
from .module import CenterFace

_Factory.register(
    cls=CenterFace,
    configs=[
        # default configuration
        CenterFaceConfig(),
    ],
)
