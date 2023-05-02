#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
In this module we define the architectures of the neural networks (models) used by the agents.
For now, we only have a `LinearModel`.
"""

import torch
import torch.nn as nn


class LinearModel(nn.Module):
    """ A very simple linear model.

    It consists of three fully connected layers:
        - input layer with ReLU activation
        - hidden layer with ReLU activation
        - output layer with softmax activation
    """
    
    def __init__(self, inputSize: int, outputSize: int, hiddenSize: int = 64) -> None:
        """ Initializes the model's layers. """
        super().__init__()
        self.fc1 = nn.Linear(inputSize, hiddenSize)
        self.fc2 = nn.Linear(hiddenSize, hiddenSize)
        self.fc3 = nn.Linear(hiddenSize, outputSize)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Performs a forward pass through the model. """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.nn.functional.softmax(self.fc3(x), dim=-1)
        return x
