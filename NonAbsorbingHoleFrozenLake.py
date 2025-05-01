# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 21:48:56 2025

@author: ghass
"""
import numpy as np
import gym
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
class NonAbsorbingHoleFrozenLake(FrozenLakeEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nrow, self.ncol = self.desc.shape
        self.nS, self.nA = self.nrow * self.ncol, self.action_space.n
        self.orig_desc = self.desc.copy()
        self._fix_hole_transitions()

    def _calc_transition(self, r, c, a):
        if a == 0:
            nc, nr = max(c-1, 0), r
        elif a == 1:
            nr, nc = min(r+1, self.nrow-1), c
        elif a == 2:
            nc, nr = min(c+1, self.ncol-1), r
        else:
            nr, nc = max(r-1, 0), c
        ns = nr * self.ncol + nc
        ltr = self.desc[nr, nc]
        return ns, float(ltr == b'G'), (ltr == b'G')

    def _calc_slip(self, r, c, a):
        return [
            (0.8, *self._calc_transition(r, c, a)),
            (0.1, *self._calc_transition(r, c, (a-1) % 4)),
            (0.1, *self._calc_transition(r, c, (a+1) % 4)),
        ]

    def _fix_hole_transitions(self):
        for s in range(self.nS):
            r, c = divmod(s, self.ncol)
            if self.orig_desc[r, c] == b'H':
                for a in range(self.nA):
                    self.P[s][a] = self._calc_slip(r, c, a)

    def step(self, a):
        out = super().step(a)
        if len(out) == 5:
            obs, rew, done_flag, truncated, info = out
            done = done_flag or truncated
        else:
            obs, rew, done, info = out
        r, c = divmod(obs, self.ncol)
        if self.orig_desc[r, c] == b'H':
            done = False
        return obs, rew, done, info

