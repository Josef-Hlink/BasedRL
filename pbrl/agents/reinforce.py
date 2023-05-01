# -*- coding: utf-8 -*-

from time import perf_counter

from pbrl.environment import CatchEnvironment
from pbrl.agents.transitions import Transition, TransitionBatch 

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb


class Network(nn.Module):
    
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 64) -> None:
    
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=-1)

        self.loss = nn.MSELoss()
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x


class REINFORCEAgent:

    def __init__(self,
        alpha: float, beta: float, gamma: float, delta: float,
        R: bool = False, V: bool = False, D: bool = False, S: bool = False,
        device: torch.device = torch.device('cpu')
    ) -> None:
        """
        Initializes the agent by setting hyperparameters and creating the network and optimizer.

        ### Args

        Hyperparameters:
            `float` alpha: learning rate
            `float` beta: entropy regularization coefficient
            `float` gamma: discount factor
            `float` delta: learning rate decay rate
        
        Interface flags:
            `bool` R: render environment
            `bool` V: verbose
            `bool` D: debug

        Misc:
            `torch.device` device: device to run on (defaults to CPU)
        """
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

        self.R = R
        self.V = V
        self.D = D
        self.S = S

        self.device = device
        self.network = Network(98, 3).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.alpha)

    def train(self, env: CatchEnvironment, nEpisodes: int) -> None:
        """ Trains the agent on an environment for a given number of episodes. """

        # allows wandb to track the network's gradients, parameters, etc.
        wandb.watch(
            self.network,
            log = 'all',
            log_freq = nEpisodes // 100
        )

        # lr decay rate is called gamma here
        # not to be confused with our discount factor which is also called gamma
        scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size = nEpisodes // 100,  # fixed for simplicity
            gamma = self.delta             # yes, this looks weird but it makes sense everywhere else
        )

        rewards = []
        start = perf_counter()

        for episode in range(nEpisodes):
            
            # (re)set environment
            state: torch.Tensor = self.castState(env.reset())
            done = False

            T = TransitionBatch()

            while not done:

                # get action probability distribution
                dist = torch.distributions.Categorical(self.network(state))
                # sample action
                action: int = dist.sample().item()
                # take action
                next_state, reward, done, _ = env.step(action)
                next_state = self.castState(next_state)
                
                T.add(Transition(state, action, reward, next_state, done))

                if self.D:
                    print(f'action dist.: {dist.probs}, action: {action}, reward: {reward}, done: {done}')

                state = next_state

            # update policy
            avgLoss = self.learn(T)
            rewards.append(T.totalReward)

            # update learning rate
            scheduler.step()

            if self.V:
                status = f'episode {episode + 1} / {nEpisodes}: reward: {T.totalReward:.2f}'
                print(f'\r{status}' + ' ' * (79-len(status)), end='', flush=True)
                if episode == nEpisodes - 1: print()
        
            wandb.log(
                dict(
                    reward = T.totalReward,
                    lr = self.optimizer.param_groups[0]['lr'],
                    loss = avgLoss,
                ),
                step = episode
            )

        print(f'average reward: {np.mean(rewards):.3f}')
        print(f'time elapsed: {perf_counter() - start:.3f} s')
        
        if self.S: torch.save(self.network.state_dict(), 'network.pth')
        
        return
    
    def learn(self, transitionBatch: TransitionBatch) -> float:
        """
        Updates the agent's policy.
        Returns the average loss over the batch.
        """

        # unpack transition batch into tensors (and put on device)
        S, A, R, S_, D = map(lambda x: x.to(self.device), transitionBatch.unpack())

        # we can't use gradients here
        with torch.no_grad():
            # initialize target value tensor
            G = torch.zeros(len(S), dtype=torch.float32).to(self.device)
            # calculate (discounted) target values
            for i in range(len(R)):
                G[i] = sum([self.gamma**j * r for j, r in enumerate(R[i:])])
        
        totalLoss = 0

        # loop over transitions
        for s, a, g in zip(S, A, G):
            dist = torch.distributions.Categorical(self.network(s))
            loss = -dist.log_prob(a) * g
            # add entropy regularization
            loss -= self.beta * dist.entropy()
            # backprop time baby
            # zero_grad used to prevent earlier gradients from affecting the current gradient
            self.optimizer.zero_grad() 
            loss.backward()
            # update parameters
            self.optimizer.step()
            totalLoss += loss.item()
        
        return totalLoss / len(S)


    def castState(self, state: np.ndarray) -> torch.Tensor:
        """ Cast 3D state (np.array) to 1D torch tensor on the correct device. """
        return torch.tensor(state, dtype=torch.float32).flatten().to(self.device)

   