#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

from pbrl.environment import CatchEnvironment
from pbrl.agents.base import PBAgent
from pbrl.agents.transitions import TransitionBatch
from pbrl.utils import ProgressBar

import numpy as np
import torch
import wandb


class ActorCriticAgent(PBAgent):
    """ An implementation of the Actor-Critic algorithm. """

    def __init__(self,
        alpha: float, beta: float, gamma: float, delta: float, batchSize: int,
        bootstrap: bool, baselineSubtraction: bool,
        device: torch.device, actor: torch.nn.Module, critic: torch.nn.Module = None,
    ) -> None:
        assert critic is not None, 'Critic cannot be None for ActorCriticAgent'
        super().__init__(alpha, beta, gamma, delta, batchSize, bootstrap, baselineSubtraction, device, actor, critic)
        return

    ##########
    # PUBLIC #
    ##########

    def train(self,
        env: CatchEnvironment, budget: int,
        Q: bool = False, D: bool = False, W: bool = False, T: bool = False,
    ) -> int:
        return super().train(env, budget, Q, D, W, T)
    
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
        """ Sets up the necessary variables for the training loop.

        1. Initializes some private variables
        2. Tells wandb to track the models if W and T are true
        3. Initializes the episode iterator
        4. Initializes the optimizers and learning rate schedulers

        ### Args
        `int` nEpisodes: number of episodes to train for
        `bool` Q: whether to use a progress bar or not
        `bool` W: whether to log anything with wandb
        `bool` T: whether to track the model with wandb
        """
        
        # 1. initialize some private variables
        super()._initTrain(nEpisodes, Q, W, T)
        
        # 2. tell wandb to track the models if W and T are true
        if W and T: wandb.watch((self.actor, self.critic), log='all', log_freq=self._uI)

        # 3. initialize the episode iterator
        if Q: self.iterator = range(self._mE)
        else: self.iterator = ProgressBar(self._mE, updateInterval=self._uI, metrics=['r', 'pg', 'vl'])
        
        # 4. initialize the optimizers and learning rate schedulers
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

        self.nSteps = 3

        return

    def _sampleEpisode(self, env: CatchEnvironment, R: bool = False) -> TransitionBatch:
        return super()._sampleEpisode(env, R)

    def _chooseAction(self, state: np.ndarray) -> int:
        return super()._chooseAction(state)

    def _learn(self, tB: TransitionBatch) -> tuple[float, float | None]:
        """Performs a learning step using the given transition batch.

        One transition batch generally represents one episode.

        ### Args
        - `TransitionBatch` tB: variable sized batch of transitions to learn from

        ### Returns
        - `float` avgPG: average policy gradient over the batch
          `float` avgVL: average value loss over the batch
        """

        # unpack transition batch into tensors (and put on device)
        S, A, R, S_, D = map(lambda x: x.to(self.device), (tB.S, tB.A, tB.R, tB.S_, tB.D))
        
        if self.bootstrap:
            G = self._getBootstrapTargets(S, A, R, S_, D)
        if self.bootstrap == False:
            with torch.no_grad():
                V_ = self.critic(S_).squeeze()
                G = R + self.gamma * V_ * (1 - D)

        totalPG, totalVL = 0, 0

        # loop over transitions with batch size
        for i in range(0, len(tB), self.batchSize):
            # slice mini-batch from the full batch
            slc: slice = slice(i, i + self.batchSize)
            _S, _A, _G = map(lambda x: x[slc], (S, A, G))
            # update policy and value function and get batch metrics
            batchPG, batchVL = self._updatePolicy(_S, _A, _G)
            
            # add batch metrics to total metrics
            totalPG += batchPG
            totalVL += batchVL

        # update learning rates
        self.aScheduler.step()
        self.cScheduler.step()

        # return average metrics
        return totalPG / len(tB), totalVL / len(tB)


    def _updatePolicy(self, S: torch.Tensor, A: torch.Tensor, G: torch.Tensor) -> tuple[float, float | None]:
        """ Updates the agent's policy and value function.

        ### Args
        `torch.Tensor` S: state tensor
        `torch.Tensor` A: action tensor
        `torch.Tensor` G: target value tensor

        ### Returns
        `float` totalPG: total policy gradient over the batch
        `float` totalVL: total value loss over the batch
        """
        maxGradNorm = 0.5
        # Baseline subtraction flag
        # self.baseSub = True

        # forward pass to get action distribution
        dist = torch.distributions.Categorical(self.actor(S))

        if self.baseSub:
            # calculate the baseline as the mean of the value function estimates.
            pred = self.critic(S).squeeze()
            baseline = pred.mean()

            # calculate advantage by using baseline subtraction.
            advantage = G - pred + baseline

            # calculate policy loss with baseline subtraction. 
            policyGradient = -dist.log_prob(A) * advantage

        else:
            # forward pass to get action distribution
            dist = torch.distributions.Categorical(self.actor(S))
            # calculate policy loss
            policyGradient = -dist.log_prob(A) * (G - self.critic(S).squeeze())

        # add entropy regularization
        policyGradient -= self.beta * dist.entropy()

        # zero_grad used to prevent earlier gradients from affecting the current gradient
        self.aOptimizer.zero_grad()
        # calculate gradients for policy network
        policyGradient.mean().backward()
        # clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), maxGradNorm)
        # update policy network parameters
        self.aOptimizer.step()

        # calculate value function loss
        if self.baseSub:
            valueLoss = ( self.critic(S).squeeze() - (G - baseline).detach()) ** 2
        else:
            valueLoss = (self.critic(S).squeeze() - G) ** 2

        # zero_grad used to prevent earlier gradients from affecting the current gradient
        self.cOptimizer.zero_grad()
        # calculate gradients for value function network
        valueLoss.mean().backward()
        # clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), maxGradNorm)
        # update value function network parameters
        self.cOptimizer.step()

        # return total policy gradient and total value loss
        return policyGradient.sum().item(), valueLoss.sum().item()

    def _getBootstrapTargets(self, S: torch.Tensor, A: torch.Tensor, R: torch.Tensor, S_: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
        """ Calculates the target values for the n-step bootstrap algorithm.

        ### Args
        `torch.Tensor` S: state tensor
        `torch.Tensor` A: action tensor
        `torch.Tensor` R: reward tensor
        `torch.Tensor` S_: next state tensor
        `torch.Tensor` D: done tensor

        ### Returns
        `torch.Tensor` G: target value tensor
        """
        # calculate target values using critic network
        with torch.no_grad():
            # estimate next state value using critic network
            V_ = self.critic(S_).squeeze()

        # calculate n-step return
        G = torch.zeros(len(S), device=self.device)
        for i in range(len(S)):
            # slice out the next n_steps transitions
            n_slc: slice = slice(i, i + self.nSteps)
            R_n, D_n, V_n = R[n_slc], D[n_slc], V_[n_slc]

            G[i] = sum([self.gamma ** k * R_n[k] for k in range(len(R_n))]) + \
                (self.gamma ** self.nSteps) * V_n[-1] * (1 - D_n[-1])

        return G

    def _logEpisode(self, i: int, r: float, pg: float, vl: float | None) -> None:
        """ Handles all logging after an episode. """
        super()._logEpisode(i, r, pg, vl)
        if not self._W: return
        lrActor = self.aOptimizer.param_groups[0]['lr']
        lrCritic = self.cOptimizer.param_groups[0]['lr']
        data = dict(reward=r, polGrad=pg, valLoss=vl, lrActor=lrActor, lrCritic=lrCritic)
        commit: bool = i % self._uI == 0
        wandb.log(data, step=i, commit=commit)
        return

    def _logFinal(self) -> None:
        """ Handles all logging after training. """
        super()._logFinal()
        if not self._Q: print(f'avg. value loss: {self._tVL / self._mE:.3f}')
        return        

    def _castState(self, state: np.ndarray | torch.Tensor) -> torch.Tensor:
        return super()._castState(state)
    
    def _checkConvergence(self, reward: float) -> None:
        return super()._checkConvergence(reward)
