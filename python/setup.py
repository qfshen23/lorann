#!/usr/bin/python

import os
import sys
import codecs
from pathlib import Path

import setuptools
from setuptools import Extension
from setuptools.command.build_ext import build_ext

parent = Path(__file__).parents[1]

with codecs.open(parent / "README.md", encoding="utf-8") as f:
    long_description = f.read()

source_files = ["lorannmodule.cc"]
ext_modules = [
    Extension(
        "lorannlib",
        source_files,
        language="c++17",
    )
]


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options.

    Assume that C++17 is available.
    """

    c_opts = {
        "unix": [
            "-std=c++17",
            "-O3",
            "-fPIC",
            "-flax-vector-conversions",
            "-DNDEBUG",
            "-DEIGEN_DONT_PARALLELIZE",
            "-Wall",
            "-Wextra",
            "-Wno-missing-field-initializers",
            "-Wno-unused-parameter",
            "-Wno-dangling-reference",
            "-Wno-c99-designator",
            "-Wno-vla-cxx-extension",
            "-Wl,--no-undefined",
        ],
    }
    libraries_opt = {
        "unix": ["stdc++"],
    }

    if sys.platform != "darwin":
        c_opts["unix"] += ["-march=native"]

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])

        import numpy as np

        for ext in self.extensions:
            ext.libraries.extend(self.libraries_opt.get(ct, []))
            ext.language = "c++17"
            ext.extra_compile_args.extend(opts)
            ext.include_dirs.extend(
                [
                    os.path.join(parent, "lorann"),
                    # Path to numpy headers
                    np.get_include(),
                ]
            )

        build_ext.build_extensions(self)


setuptools.setup(
    name="lorann",
    author="Elias Jääsaari",
    author_email="elias.jaasaari@gmail.com",
    version="0.1",
    description="Fast and lightweight library for approximate nearest neighbor search",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ejaasaari/lorann",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="vector search, approximate nearest neighbor search",
    packages=setuptools.find_packages(),
    zip_safe=False,
    test_suite="py.test",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
)
