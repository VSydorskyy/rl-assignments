import copy
from pathlib import Path
import random
import math

from gym.spaces import Discrete
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(object):
    def __init__(
        self,
        value_network,
        action_space,
        start_eps, final_eps, decay_eps,
        batch_size,
        learning_rate,
        discount_factor,
        update_target_rate,
        cuda=False
    ):
        self.value_network = value_network
        self.target_network = copy.deepcopy(value_network)
        self.start_eps = start_eps
        self.final_eps = final_eps
        self.decay_eps = decay_eps
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.update_target_rate = update_target_rate
        self.cuda = cuda

        self.global_step = 0

        assert isinstance(action_space, Discrete), \
            "Action space has to be discrete"

        self.action_space = action_space

        if cuda:
            self.value_network.cuda()
            self.target_network.cuda()

        self.optimizer = torch.optim.Adam(self.value_network.parameters(), lr=learning_rate)

        if cuda:
          self.optimizer.cuda()  

    def update_target(self):
        self.target_network.load_state_dict(self.value_network.state_dict())

    def train_step(self, replay_buffer):
        batch = replay_buffer.sample(self.batch_size)
        v_s0, v_a, v_s1, v_r, v_d = zip(*batch)

        if self.cuda:
            FloatTensor = torch.cuda.FloatTensor
            LongTensor = torch.cuda.LongTensor
        else:
            FloatTensor = torch.FloatTensor
            LongTensor = torch.LongTensor

        v_s0 = FloatTensor(v_s0)
        v_a = LongTensor(v_a)
        v_s1 = FloatTensor(v_s1)
        v_r = FloatTensor(v_r)
        v_d = FloatTensor(v_d)

        y_hat = self.value_network(v_s0).gather(1, v_a.unsqueeze(1)).squeeze(-1)
        y = self.target_network(v_s1).detach().max(1)[0]*(1-v_d) + v_r

        loss = F.smooth_l1_loss(y_hat, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.global_step % self.update_target_rate == 0:
            self.update_target()

        self.global_step += 1

        return loss.detach().cpu().item()

    def prediction_step(self, s, force_greedy=False):
        if self.cuda:
            s = s.cuda()

        sample = random.random()
        eps_threshold = self.start_eps + (self.start_eps - self.final_eps) * math.exp(-1. * self.global_step / self.decay_eps)

        if force_greedy:
            action =  self.value_network(s).argmax(1).item()
        else:
            if sample > eps_threshold:
                action = self.value_network(s).argmax(1).item()
            else:
                action = self.action_space.sample()

        return action

