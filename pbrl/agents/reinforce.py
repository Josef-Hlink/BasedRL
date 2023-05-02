#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

from pbrl.environment import CatchEnvironment
from pbrl.agents.transitions import Transition, TransitionBatch 

import numpy as np
import torch
import wandb


class REINFORCEAgent:

    def __init__(self,
        model: torch.nn.Module, device: torch.device,
        alpha: float, beta: float, gamma: float, delta: float,
        V: bool = False, D: bool = False, W: bool = False
    ) -> None:
        """ Initializes the agent by setting hyperparameters and creating the model and optimizer.

        ### Args

        Torch:
            `torch.nn.Module` model: the model to be used as the agent's function approximator
            `torch.device` device: device to run on

        Hyperparameters:
            `float` alpha: learning rate
            `float` beta: entropy regularization coefficient
            `float` gamma: discount factor
            `float` delta: learning rate decay rate
        
        Flags:
            `bool` V: toggle verbose printing
            `bool` D: toggle debug printing
            `bool` W: toggle wandb logging
        """
        
        self.device = device
        self.model = model.to(self.device)

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

        self.V = V
        self.D = D
        self.W = W

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.alpha)
        return

    ##########
    # PUBLIC #
    ##########

    def train(self, env: CatchEnvironment, nEpisodes: int) -> None:
        """ Trains the agent on an environment for a given number of episodes. """

        # allows wandb to track the model's gradients, parameters, etc.
        if self.W:
            wandb.watch(
                self.model,
                log = 'all',
                log_freq = nEpisodes // 100
            )

        # lr decay rate is called gamma here
        # not to be confused with our discount factor which is also called gamma
        scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size = nEpisodes // 100,  # fixed for simplicity
            gamma = self.delta             # yes, this looks weird but it makes sense everywhere else
        )

        for episode in range(nEpisodes):
            
            # (re)set environment
            state: torch.Tensor = self._castState(env.reset(), flatten=True)
            done = False

            # (re)init transition batch
            T = TransitionBatch()

            while not done:

                # choose action
                action: int = self.chooseAction(state)
                # take action
                next_state, reward, done, _ = env.step(action)
                next_state = self._castState(next_state, flatten=True)
                
                T.add(Transition(state, action, reward, next_state, done))

                state = next_state

            # update policy and learning rate
            avgLoss = self._learn(T)
            scheduler.step()

            # log to console
            if self.V:
                status = f'episode {episode + 1} / {nEpisodes}: reward: {T.totalReward:.2f}'
                print(f'\r{status}' + ' ' * (79-len(status)), end='', flush=True)
        
            # log to wandb
            if self.W:
                wandb.log(
                    dict(
                        reward = T.totalReward,
                        lr = self.optimizer.param_groups[0]['lr'],
                        loss = avgLoss,
                    ),
                    step = episode
                )

        if self.V:
            print('\ntraining complete\n' + '-' * 79)

        return
    
    def saveModel(self, path: Path) -> None:
        """ Saves the model's parameters to the given path. """
        torch.save(self.model.state_dict(), path)
        return
    
    def loadModel(self, path: Path) -> None:
        """ Loads the model's parameters from the given path. """
        self.model.load_state_dict(torch.load(path))
        return
    
    def chooseAction(self, state: np.ndarray) -> int:
        """ Chooses an action to take based on the current policy. """
        state = self._castState(state, flatten=True)
        return torch.distributions.Categorical(self.model(state)).sample().item()

    ###########
    # PRIVATE #
    ###########

    def _learn(self, transitionBatch: TransitionBatch) -> float:
        """ Updates the agent's policy.

        ### Args
        `TransitionBatch` transitionBatch: variable sized batch of transitions to learn from

        ### Returns
        `float` avgLoss: average loss over the batch
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
            dist = torch.distributions.Categorical(self.model(s))
            loss = -dist.log_prob(a) * g
            # add entropy regularization
            loss -= self.beta * dist.entropy()
            # zero_grad used to prevent earlier gradients from affecting the current gradient
            self.optimizer.zero_grad() 
            loss.backward()
            # update parameters
            self.optimizer.step()
            totalLoss += loss.item()
        
        return totalLoss / len(S)

    def _castState(self, state: np.ndarray | torch.Tensor, flatten: bool = False) -> torch.Tensor:
        """
        Cast numpy array or a torch tensor to a torch tensor on the correct device.
        If `flatten`, the tensor is cast to a 1D tensor.
        """
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(self.device)
        if flatten:
            state = state.flatten()
        return state
