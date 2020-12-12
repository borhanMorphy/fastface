import importlib
import archs
import os

__ADAPTERS__ = ('gdrive',)

def handle_model_download(file_path:str, arch_name:str, config:str):
    if os.path.exists(file_path): return

    arch_configs = archs.get_arch_config_by_name(arch_name, config=config)
    assert "adapter" in arch_configs,"architecture does not contain adapter"
    adapter_configs = arch_configs.pop('adapter')
    assert adapter_configs['type'] in __ADAPTERS__,f"{adapter_configs['type']} not in the supported adapters"

    key = adapter_configs['key']
    if not file_path.endswith(key): return

    os.makedirs(os.path.dirname(key), exist_ok=True)

    adapter = importlib.import_module(f"registry.adapters.{adapter_configs['type']}")

    try:
        adapter.Adapter.download(key,
            *adapter_configs['args'],**adapter_configs['kwargs'])
    except Exception as e:
        # TODO add logging
        raise AssertionError(f"download failed with: {e}")