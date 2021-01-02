import pytest
import mypackage

"""
mypackage.utils.cache.get_cache_path
mypackage.utils.cache.get_model_cache_path
mypackage.utils.cache.get_data_cache_path

mypackage.utils.config.get_pkg_root_path
mypackage.utils.config.get_pkg_arch_path
mypackage.utils.config.get_registry_path
mypackage.utils.config.get_registry
mypackage.utils.config.discover_archs
mypackage.utils.config.get_arch_pkg
mypackage.utils.config.get_arch_cls

mypackage.utils.utils.seed_everything
mypackage.utils.utils.random_sample_selection
mypackage.utils.utils.get_best_checkpoint_path
mypackage.utils.visualize.prettify_detections
"""


@pytest.mark.parametrize("func",
    [
        "get_cache_path","get_model_cache_path",
        "get_data_cache_path"
    ]
)
def test_cache_func_exists(func:str):
    assert func in dir(mypackage.utils.cache),f"{func} not found in the mypackage.utils.cache"

@pytest.mark.parametrize("func",
    [
        "get_pkg_root_path", "get_pkg_arch_path",
        "get_registry_path", "get_registry",
        "discover_archs", "get_arch_pkg","get_arch_cls"
    ]
)
def test_config_func_exists(func:str):
    assert func in dir(mypackage.utils.config),f"{func} not found in the mypackage.utils.config"


@pytest.mark.parametrize("func",
    [
        "seed_everything","random_sample_selection",
        "get_best_checkpoint_path"
    ]
)
def test_utils_func_exists(func:str):
    assert func in dir(mypackage.utils.utils),f"{func} not found in the mypackage.utils.utils"


@pytest.mark.parametrize("func",["prettify_detections"])
def test_visualize_func_exists(func:str):
    assert func in dir(mypackage.utils.visualize),f"{func} not found in the mypackage.utils.visualize"