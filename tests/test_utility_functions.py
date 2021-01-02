import pytest
import fastface

@pytest.mark.parametrize("func",
    [
        "get_cache_path","get_model_cache_path",
        "get_data_cache_path","get_checkpoint_cache_path"
    ]
)
def test_cache_func_exists(func:str):
    assert func in dir(fastface.utils.cache),f"{func} not found in the fastface.utils.cache"

@pytest.mark.parametrize("func",
    [
        "get_pkg_root_path", "get_pkg_arch_path",
        "get_registry_path", "get_registry",
        "discover_archs", "get_arch_pkg","get_arch_cls"
    ]
)
def test_config_func_exists(func:str):
    assert func in dir(fastface.utils.config),f"{func} not found in the fastface.utils.config"


@pytest.mark.parametrize("func",
    [
        "seed_everything","random_sample_selection"
    ]
)
def test_random_func_exists(func:str):
    assert func in dir(fastface.utils.random),f"{func} not found in the fastface.utils.random"


@pytest.mark.parametrize("func",["prettify_detections"])
def test_visualize_func_exists(func:str):
    assert func in dir(fastface.utils.visualize),f"{func} not found in the fastface.utils.visualize"