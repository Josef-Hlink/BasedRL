#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name = 'pbrl',
    version = '0.0.1',
    description = 'Policy-based Reinforcement Learning',
    author = 'Josef Hamelink',
    license = 'MIT',
    packages = find_packages(),
    install_requires = [
        'numpy',
        'pandas',
        'scipy',
        'scikit-learn',
        'matplotlib',
        'ipykernel',
        'jupyter',
        'gym',
        'wandb',
    ],
    entry_points = {
        'console_scripts': [
            'pbrl = pbrl.cli:main',
        ],
    },
)
