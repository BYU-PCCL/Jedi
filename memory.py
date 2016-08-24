"""Code from https://github.com/tambetm/simple_dqn/blob/master/src/replay_memory.py"""
import random
import numpy as np
from threading import Lock
from collections import deque

class Memory:
    def __init__(self, args, environment):
        self.args = args
        self.dims = environment.get_state_space()

        self.actions = np.empty(tuple([args.replay_memory_size]) + environment.get_action_space(), dtype=environment.get_action_datatype())
        self.rewards = np.empty(args.replay_memory_size, dtype=np.float64)
        self.priorities = np.zeros(args.replay_memory_size, dtype=np.float64)
        self.screens = np.empty(tuple([args.replay_memory_size]) + self.dims, dtype=environment.get_state_datatype())
        self.terminals = np.empty(args.replay_memory_size, dtype=np.bool)

        self.count = 0
        self.current = 0
        self.priority_sum = 0

        self.void_phi = np.zeros(tuple([args.phi_frames]) + self.dims, dtype=environment.get_state_datatype())

        # start a thread

    def update(self, index, priority):
        if self.args.use_prioritization:
            for si, i in enumerate(index):
                self.priority_sum -= self.priorities[i]
                self.priorities[i] = priority[si]
                self.priority_sum += priority[si]

    def add(self, screen, reward, action, terminal, priority=100):
        assert screen.shape == self.dims
        # NB! screen is post-state, after action and reward
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.screens[self.current, ...] = screen
        self.terminals[self.current] = terminal
        self.update([self.current], [priority])

        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.args.replay_memory_size

    def get_state(self, index):
        assert self.count > 0, "replay memory is empty, use at least --random_steps 1"
        # normalize index to expected range, allows negative indexes
        index = index % self.count
        # if is not in the beginning of matrix
        if index >= self.args.phi_frames - 1:
            # use faster slicing
            return self.screens[(index - (self.args.phi_frames - 1)):(index + 1), ...]
        else:
            # otherwise normalize indexes and use slower list based access
            indexes = [(index - i) % self.count for i in reversed(range(self.args.phi_frames))]
            return self.screens[indexes, ...]

    def can_sample(self): 
        return self.count > self.args.batch_size

    def sample_priority_indexes(self, size):
        cumsum = np.cumsum(self.priorities[0:self.count])
        proposal = np.random.random(size) * cumsum[-1]
        # Faster than np.random.choice
        return list(np.searchsorted(cumsum, proposal, side='left'))

    def sample(self):
        # memory must include poststate, prestate and history
        assert self.can_sample()

        indexes = []
        random_indexes = []

        prestates = np.zeros((self.args.batch_size, self.args.phi_frames) + self.dims, dtype=self.screens.dtype)
        poststates = np.zeros((self.args.batch_size, self.args.phi_frames) + self.dims, dtype=self.screens.dtype)
        lookaheads = np.zeros((self.args.batch_size, self.args.phi_frames) + self.dims, dtype=self.screens.dtype)

        if self.args.use_prioritization:
            random_indexes = self.sample_priority_indexes(self.args.batch_size)

        while len(indexes) < self.args.batch_size:
            # find random index
            while True:
                # sample one index (ignore states wraping over
                if len(random_indexes) > 0:
                    index = random_indexes.pop()
                else:
                    index = random.randint(self.args.phi_frames, self.count - 1)

                # if wraps over current pointer, then get new one
                if index >= self.current and index - self.args.phi_frames < self.current:
                    continue
                # if wraps over episode end, then get new one
                # NB! poststate (last screen) can be terminal state!
                if self.terminals[(index - self.args.phi_frames):index].any():
                    continue
                # otherwise use this index
                break

            # NB! having index first is fastest in C-order matrices
            prestates[len(indexes), ...] = self.get_state(index - 1)
            poststates[len(indexes), ...] = self.get_state(index)

            # if lookahead spans into the next episode, void it
            if self.terminals[index:index + self.args.lookahead].any():
                lookaheads[len(indexes), ...] = self.void_phi
            else:
                lookaheads[len(indexes), ...] = self.get_state(index + self.args.lookahead)

            indexes.append(index)

        actions = self.actions[indexes]
        rewards = self.rewards[indexes]
        terminals = self.terminals[indexes]

        return prestates.copy(), actions, rewards, poststates.copy(), terminals, lookaheads.copy(), indexes