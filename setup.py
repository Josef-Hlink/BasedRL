#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name = 'pbrl',
    version = '0.1.0',
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
            'pbrl-run = pbrl.run:main',
            'pbrl-render = pbrl.render:main',
            'pbrl-sweep = pbrl.sweep:main',
        ],
    },
)
