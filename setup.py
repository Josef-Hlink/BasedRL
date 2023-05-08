#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name = 'pbrl',
    version = '1.0.0',
    description = 'Policy-based Reinforcement Learning',
    author = 'Josef Hamelink',
    license = 'MIT',
    packages = find_packages(),
    install_requires = [
        'wheel',
        'numpy',
        'pandas',
        'scipy',
        'scikit-learn',
        'matplotlib',
        'ipykernel',
        'jupyter',
        'gym',
        'wandb',
        'torch',
    ],
    entry_points = {
        'console_scripts': [
            'pbrl-run = pbrl.cli.run:main',
            'pbrl-render = pbrl.cli.render:main',
            'pbrl-sweep = pbrl.cli.sweep:main',
        ],
    },
)
