import pytest

import fastface as ff

# TODO expand here


@pytest.mark.parametrize(
    "func",
    [
        "get_cache_dir",
        "get_model_cache_dir",
        "get_data_cache_dir",
        "get_checkpoint_cache_dir",
    ],
)
def test_cache_func_exists(func: str):
    assert func in dir(
        ff.utils.cache
    ), "{} not found in the fastface.utils.cache".format(func)


@pytest.mark.parametrize(
    "func",
    [
        "get_pkg_root_path",
        "get_pkg_arch_path",
        "get_registry_path",
        "get_registry",
        "discover_archs",
        "get_arch_pkg",
        "get_arch_cls",
    ],
)
def test_config_func_exists(func: str):
    assert func in dir(
        ff.utils.config
    ), "{} not found in the fastface.utils.config".format(func)


@pytest.mark.parametrize("func", ["render_predictions"])
def test_visualize_func_exists(func: str):
    assert func in dir(ff.utils.vis), "{} not found in the fastface.utils.vis".format(
        func
    )
