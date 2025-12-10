import os
import subprocess
import sys

from setuptools import Extension, find_packages, setup
import numpy as np

from build_helpers.build_helpers import (
    ANTLRCommand,
    HYDRAANTLRCommand,
    BuildPyCommand,
    CleanCommand,
    DevelopCommand,
    SDistCommand,
    find_version,
)


if sys.platform == "darwin":
    extra_compile_args = ["-stdlib=libc++", "-O3"]
else:
    extra_compile_args = ["-std=c++11", "-O3"]


class NumpyExtension(Extension):
    """Source: https://stackoverflow.com/a/54128391"""

    def __init__(self, *args, **kwargs):
        self.__include_dirs = []
        super().__init__(*args, **kwargs)

    @property
    def include_dirs(self):
        import numpy

        return self.__include_dirs + [numpy.get_include()]

    @include_dirs.setter
    def include_dirs(self, dirs):
        self.__include_dirs = dirs

extensions = [
    Extension(
        "fairseq.libbleu",
        sources=[
            "fairseq/clib/libbleu/libbleu.cpp",
            "fairseq/clib/libbleu/module.cpp",
        ],
        extra_compile_args=extra_compile_args,
    ),
    NumpyExtension(
        "fairseq.data.data_utils_fast",
        sources=["fairseq/data/data_utils_fast.pyx"],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
    NumpyExtension(
        "fairseq.data.token_block_utils_fast",
        sources=["fairseq/data/token_block_utils_fast.pyx"],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
]

setup(
    cmdclass={
        "antlr": ANTLRCommand,
        "hydra_antlr":HYDRAANTLRCommand,
        "clean": CleanCommand,
        "sdist": SDistCommand,
        "build_py": BuildPyCommand,
        "develop": DevelopCommand,
    },
    name="fairseq_extension",
    ext_modules=extensions,
)