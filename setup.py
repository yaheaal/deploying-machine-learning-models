#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from setuptools import find_packages, setup

# Package meta-data.
NAME = 'classification_model'
DESCRIPTION = "A sample Python project for deploying machine learning models, focusing on classification tasks."
URL = "https://github.com/yaheaal/deploying-machine-learning-models"
EMAIL = "yaheaal@hotmail.com"
AUTHOR = "Yahea Al"
REQUIRES_PYTHON = ">=3.7"

# Load the package's version from a VERSION file
ROOT_DIR = Path(__file__).resolve().parent
REQUIREMENTS_DIR = ROOT_DIR / 'requirements'
PACKAGE_DIR = ROOT_DIR / 'classification_model'  
with open(PACKAGE_DIR / "VERSION", 'r', encoding='utf-8') as f:
    _version = f.read().strip()

# Load requirements from a requirements.txt file
def list_reqs(fname='requirements.txt'):
    with open(REQUIREMENTS_DIR / fname, 'r', encoding='utf-8') as fd:
        return fd.read().splitlines()

# Package setup
setup(
    name=NAME,
    version=_version,
    description=DESCRIPTION,
    long_description=open('README.md', 'r', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests"]),
    package_data={"classification_model": ["VERSION"]},
    install_requires=list_reqs(),
    extras_require={},
    include_package_data=True,
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
)
