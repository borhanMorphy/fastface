from ..module import Detector

"""
- from_pretrained(model:str) -> pl.LightninModule
- from_checkpoint(arch:str, ckpt_path:str) -> pl.LightningModule
- build(arch:str, config:Union[str,Dict]) -> pl.LightningModule

- list_models() -> List[str]
- list_archs() -> List[str]
- list_arch_configs(arch:str) -> List[str]

- get_arch_config(arch:str, config:str) -> Dict
"""