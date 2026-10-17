"""Packaging checks: public surface, version and typing marker."""

import importlib.resources
import tomllib
from pathlib import Path

import radix_kv as rk


def test_public_names_are_exported() -> None:
    for name in rk.__all__:
        assert hasattr(rk, name), name


def test_version_matches_the_cmake_project() -> None:
    cmake = Path(__file__).resolve().parents[2] / "CMakeLists.txt"
    if cmake.exists():  # running from a source checkout
        assert f"VERSION {rk.__version__}" in cmake.read_text(encoding="utf-8")
    assert rk.__version__.count(".") == 2


def test_package_ships_type_information() -> None:
    files = importlib.resources.files("radix_kv")
    assert files.joinpath("py.typed").is_file()
    assert files.joinpath("_core.pyi").is_file()


def test_pyproject_declares_python_floor() -> None:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if pyproject.exists():
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        assert data["project"]["requires-python"] == ">=3.11"
