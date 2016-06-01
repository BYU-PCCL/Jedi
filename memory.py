"""Code from https://github.com/tambetm/simple_dqn/blob/master/src/replay_memory.py"""
import random
import numpy as np

class Memory:
    def __init__(self, args, environment):
        self.args = args
        self.dims = environment.get_state_space()

        self.actions = np.empty(args.replay_memory_size, dtype=np.uint8)
        self.rewards = np.empty(args.replay_memory_size, dtype=np.integer)
        self.screens = np.empty(tuple([args.replay_memory_size]) + self.dims, dtype=np.uint8)
        self.terminals = np.empty(args.replay_memory_size, dtype=np.bool)

        self.count = 0
        self.current = 0

        # pre-allocate prestates and poststates for minibatch
        self.prestates = np.empty((args.batch_size, args.phi_frames) + self.dims, dtype=np.uint8)
        self.poststates = np.empty((args.batch_size, args.phi_frames) + self.dims, dtype=np.uint8)

    def add(self, screen, reward, action, terminal):
        assert screen.shape == self.dims
        # NB! screen is post-state, after action and reward
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.screens[self.current, ...] = screen
        self.terminals[self.current] = terminal

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
        return self.count > self.args.phi_frames

    def sample(self):
        # memory must include poststate, prestate and history
        assert self.count > self.args.phi_frames
        # sample random indexes
        indexes = []
        while len(indexes) < self.args.batch_size:
            # find random index
            while True:
                # sample one index (ignore states wraping over
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
            self.prestates[len(indexes), ...] = self.get_state(index - 1)
            self.poststates[len(indexes), ...] = self.get_state(index)
            indexes.append(index)

        actions = self.actions[indexes]
        rewards = self.rewards[indexes]
        terminals = self.terminals[indexes]

        # todo: obviously not a real lookahead
        lookaheads = self.poststates.copy()

        return self.prestates.copy(), actions, rewards, self.poststates.copy(), terminals, lookaheads, indexes