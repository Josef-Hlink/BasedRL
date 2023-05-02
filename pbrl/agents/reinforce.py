#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

from pbrl.environment import CatchEnvironment
from pbrl.agents.transitions import Transition, TransitionBatch 
from pbrl.utils import ProgressBar

import numpy as np
import torch
import wandb


class REINFORCEAgent:

    def __init__(self,
        model: torch.nn.Module, device: torch.device,
        alpha: float, beta: float, gamma: float, delta: float,
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
        """
        
        self.device = device
        self.model = model.to(self.device)

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.alpha)
        self.convergeCounter: int = 0
        return

    ##########
    # PUBLIC #
    ##########

    def train(self,
        env: CatchEnvironment, nEpisodes: int,
        V: bool = False, D: bool = False, W: bool = False,
    ) -> None:
        """ Trains the agent on an environment for a given number of episodes.
        
        ### Args
        `CatchEnvironment` env: environment to train on
        `int` nEpisodes: number of episodes to train for
        `bool` V: toggle verbose printing
        `bool` D: toggle debug printing
        `bool` W: toggle wandb logging
        """

        # initialize training
        self._initTrain(nEpisodes, V, W)

        for i in self.iterator:

            # sample episode and get reward
            TB = self._sampleEpisode(env)
            episodeR = TB.totalReward
            
            # update policy and learning rate
            episodeL = self._learn(TB)
            self.scheduler.step()

            # log to console and wandb
            self._log(episodeR, episodeL, i=i)
            
            # check for convergence
            self._checkConvergence(episodeR)
            if self.converged:
                break
        
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

    @property
    def converged(self) -> bool:
        """ Hardcoded tolerance for now. """
        return self.convergeCounter >= 50

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

    def _initTrain(self, nEpisodes: int, V: bool, W: bool) -> None:
        """ Sets up the necessary variables for the training loop.

        1. Initializes some private variables
        2. Tells wandb to track the model if W is True
        3. Initializes the episode iterator
        4. Initializes the learning rate scheduler

        ### Args
        `int` nEpisodes: number of episodes to train for
        `bool` V: whether to use the progress bar, or just use a `range` object to iterate over episodes
        `bool` W: whether to use wandb to track model
        """
        
        self._nE = nEpisodes         # total number of episodes
        self._uI = self._nE // 100   # update interval
        self._maR, self._maL = 0, 0  # moving averages for reward and loss
        self._V, self._W = V, W      # whether to use progress bar and wandb
        
        if W: wandb.watch(self.model, log='all', log_freq=self._uI)
        
        if V: self.iterator = ProgressBar(self._nE, updateInterval=self._uI, metrics=['r', 'l'])
        else: self.iterator = range(self._nE)
        
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size = self._uI,  # fixed for simplicity (TODO: fix this, haha...)
            gamma = self.delta     # yes, this looks weird but it makes sense everywhere else
        )
        
        return

    def _sampleEpisode(self, env: CatchEnvironment) -> TransitionBatch:
        """ Samples an episode from the environment and returns the sampled transitions in a `TransitionBatch`. """
        TB = TransitionBatch()
        state = self._castState(env.reset(), flatten=True)
        done = False
        while not done:
            action = self.chooseAction(state)
            next_state, reward, done, _ = env.step(action)
            next_state = self._castState(next_state, flatten=True)
            TB.add(Transition(state, action, reward, next_state, done))
            state = next_state
        return TB

    def _log(self, eR: float, eL: float, i: int) -> None:
        """ Handles all logging after an episode. """
        self._maR += eR / self._uI
        self._maL += eL / self._uI
        if (i+1) % self._uI == 0:
            if self._V:
                self.iterator.updateMetrics(r=self._maR, l=self._maL)
            else:
                status = f'episode {i + 1} / {self._nE} | r: {self._maR:.2f} | l: {self._maL:.2f}'
                print(f'\r{status}' + ' ' * (79-len(status)), end='', flush=True)
                if (i+1) == self._nE: print('\nfinished training')
            self._maR, self._maL = 0, 0
        if self._W:
            wandb.log(dict(reward=eR, lr=self.optimizer.param_groups[0]['lr'], loss=eL), step=i)
        return

    def _castState(self, state: np.ndarray | torch.Tensor, flatten: bool = False) -> torch.Tensor:
        """ Cast numpy array or a torch tensor to a torch tensor on the correct device.
        If `flatten`, the tensor is cast to a 1D tensor.
        """
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(self.device)
        if flatten:
            state = state.flatten()
        return state

    def _checkConvergence(self, reward: float) -> None:
        """ Updates the convergence counter. """
        if reward >= 30:
            self.convergeCounter += 1
        else:
            self.convergeCounter = 0
