#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pbrl.environment import CatchEnvironment

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class Network(nn.Module):
    
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 64) -> None:
    
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)

        self.loss = nn.MSELoss()
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


class REINFORCEAgent:

    def __init__(self, alpha: float, render: bool, d_print: bool) -> None:
        self.alpha = alpha
        self.network = Network(98, 3)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.alpha)
        self.render = render
        self.d_print = d_print

    def __call__(self) -> str:
        return 'hello world'

    def learn(self, env: CatchEnvironment, episodes: int = 100) -> None:
        """
        Learn for 100 episodes.
        """

        state = env.reset()
        state = cast_state(state)

        for episode in range(episodes):
            done = False
            env.reset() # reset the environment for each episode, resets done boolean. 
            while not done:

                if self.render:
                    env.render()

                pred = self.network(state)
                action = pred.argmax().item()
                next_state, reward, done, _ = env.step(action)

                if self.d_print:
                    print(f"prediction: {pred}, action: {action}, reward: {reward}, done: {done}")

                next_state = cast_state(next_state)
                self.update(state, action, reward, next_state, done)
                state = next_state

            print(f'episode {episode + 1} / {episodes} done')
        return
    
    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """
        Update the agent's policy.
        """
        target = self.network(state)
        target[action] = reward
        self.optimizer.zero_grad()
        loss = self.network.loss(target, self.network(state))
        loss.backward()
        self.optimizer.step()
        

def cast_state(state: np.ndarray) -> torch.Tensor:
    """ Cast 3D state (np.array) to 1D torch tensor. """
    return torch.tensor(state, dtype=torch.float32).flatten()
