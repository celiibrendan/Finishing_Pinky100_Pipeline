#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='funconnect',
    version='0.0.1',
    description='Datajoint schemas for funconnect analysis',
    author='Fabian Sinz',
    author_email='sinz@bcm.edu',
    packages=find_packages(exclude=[]),
    install_requires=['numpy', 'tqdm', 'gitpython', 'scikit-image', 'datajoint', 'jgraph'],
)
