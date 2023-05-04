#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

from pbrl.environment import CatchEnvironment
from pbrl.agents.base import PBAgent
from pbrl.agents.transitions import TransitionBatch

import numpy as np
import torch
import wandb


class ActorCriticAgent(PBAgent):
    """ An implementation of the Actor-Critic algorithm. """

    def __init__(self,
        alpha: float, beta: float, gamma: float, delta: float, batchSize: int,
        device: torch.device, actor: torch.nn.Module, critic: torch.nn.Module = None,
    ) -> None:
        assert critic is not None, 'Critic cannot be None for ActorCriticAgent'
        super().__init__(alpha, beta, gamma, delta, batchSize, device, actor, critic)
        return

    ##########
    # PUBLIC #
    ##########

    def train(self,
        env: CatchEnvironment, nEpisodes: int,
        Q: bool = False, D: bool = False, W: bool = False, T: bool = False,
    ) -> None:
        return super().train(env, nEpisodes, Q, D, W, T)
    
    def evaluate(self, env: CatchEnvironment, nEpisodes: int, R: bool = False) -> float:
        return super().evaluate(env, nEpisodes, R)

    @property
    def converged(self) -> bool:
        return super().converged
    
    def saveModels(self, path: Path) -> None:
        """ Saves the models' parameters to the given path. """
        torch.save(self.actor.state_dict(), path / 'actor.pth')
        torch.save(self.critic.state_dict(), path / 'critic.pth')
        return
    
    def loadModels(self, path: Path) -> None:
        """ Loads the models' parameters from the given path. """
        self.actor.load_state_dict(torch.load(path / 'actor.pth'))
        self.critic.load_state_dict(torch.load(path / 'critic.pth'))
        return

    ###########
    # PRIVATE #
    ###########

    def _initTrain(self, nEpisodes: int, Q: bool, W: bool, T: bool) -> None:

        super()._initTrain(nEpisodes, Q, W, T)
        
        if W and T: wandb.watch((self.actor, self.critic), log='all', log_freq=self._uI)

        self.aOptimizer = torch.optim.Adam(self.actor.parameters(), lr=self.alpha)
        self.cOptimizer = torch.optim.Adam(self.critic.parameters(), lr=self.alpha)

        self.aScheduler = torch.optim.lr_scheduler.StepLR(
            self.aOptimizer,
            step_size = self._uI,  # fixed for simplicity
            gamma = self.delta     # yes, this looks weird but it makes sense everywhere else
        )
        self.cScheduler = torch.optim.lr_scheduler.StepLR(
            self.cOptimizer,
            step_size = self._uI,
            gamma = self.delta
        )

        return

    def _sampleEpisode(self, env: CatchEnvironment, R: bool = False) -> TransitionBatch:
        return super()._sampleEpisode(env, R)

    def _chooseAction(self, state: np.ndarray) -> int:
        return super()._chooseAction(state)

    def _learn(self, tB: TransitionBatch) -> float:
        """ Performs a learning step using the given transition batch.

        ### Args
        `TransitionBatch` tB: variable sized batch of transitions to learn from

        ### Returns
        `float` avgLoss: average loss over the batch
        """

        # unpack transition batch into tensors (and put on device)
        S, A, R, S_, D = map(lambda x: x.to(self.device), (tB.S, tB.A, tB.R, tB.S_, tB.D))

        # we can't use gradients here
        with torch.no_grad():
            # calculate target values using critic network
            V_ = self.critic(S_).squeeze()
            G = R + self.gamma * V_ * (1 - D)

        totalLoss = 0

        # loop over transitions with batch size
        for i in range(0, len(tB), self.batchSize):
            # slice mini-batch from the full batch
            slc: slice = slice(i, i+self.batchSize)
            _S, _A, _G = map(lambda x: x[slc], (S, A, G))
            # update policy and value function and add loss to total loss
            totalLoss += self._updatePolicy(_S, _A, _G)

        # update learning rates
        self.aScheduler.step()
        self.cScheduler.step()

        # return average loss
        return totalLoss / len(tB)


    def _updatePolicy(self, S: torch.Tensor, A: torch.Tensor, G: torch.Tensor) -> float:
        """ Updates the agent's policy and value function.

        ### Args
        `torch.Tensor` S: state tensor
        `torch.Tensor` A: action tensor
        `torch.Tensor` G: target value tensor

        ### Returns
        `float` totalLoss: total loss over the batch
        """

        maxGradNorm = 0.5

        # forward pass to get action distribution
        dist = torch.distributions.Categorical(self.actor(S))
        # calculate policy loss
        policyLoss = -dist.log_prob(A) * (G - self.critic(S).squeeze())
        # add entropy regularization
        policyLoss -= self.beta * dist.entropy()

        # zero_grad used to prevent earlier gradients from affecting the current gradient
        self.aOptimizer.zero_grad()
        # calculate gradients for policy network
        policyLoss.mean().backward()
        # clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), maxGradNorm)
        # update policy network parameters
        self.aOptimizer.step()

        # calculate value function loss
        valueLoss = (self.critic(S).squeeze() - G.detach())**2

        # zero_grad used to prevent earlier gradients from affecting the current gradient
        self.cOptimizer.zero_grad()
        # calculate gradients for value function network
        valueLoss.mean().backward()
        # clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), maxGradNorm)
        # update value function network parameters
        self.cOptimizer.step()

        # return total loss over the batch
        return (policyLoss + valueLoss).sum().item()

    def _logEpisode(self, i: int, r: float, l: float) -> None:
        """ Handles all logging after an episode. """
        super()._logEpisode(i, r, l)
        if not self._W: return
        lrActor = self.aOptimizer.param_groups[0]['lr']
        lrCritic = self.cOptimizer.param_groups[0]['lr']
        data = dict(reward=r, lrActor=lrActor, lrCritic=lrCritic, loss=l)
        commit: bool = i % self._uI == 0
        wandb.log(data, step=i, commit=commit)
        return

    def _logFinal(self) -> None:
        """ Handles all logging after training. """
        return super()._logFinal()

    def _castState(self, state: np.ndarray | torch.Tensor) -> torch.Tensor:
        return super()._castState(state)
    
    def _checkConvergence(self, reward: float) -> None:
        return super()._checkConvergence(reward)
