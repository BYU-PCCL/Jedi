from __future__ import division
import random
import numpy as np
from memory import Memory
import Queue
from threading import Thread

class Agent:
    def __init__(self, args, environment, network):
        self.args = args
        self.network = network
        self.environment = environment
        self.num_actions = environment.get_num_actions()
        self.memory = Memory(args, environment)
        self.epsilon = 1
        self.iterations = 0
        self.training_queue = Queue.Queue(maxsize=args.threads)

        self.phi = np.zeros([args.phi_frames, args.resize_height, args.resize_width], dtype=np.uint8)

        self.threads = []
        for _ in range(args.threads):
            self.threads.append(Thread(target=self.train))
            self.threads[-1].setDaemon(True)
            self.threads[-1].start()

    def get_action(self, state, is_evaluate):
        self.iterations += 1
        if random.random() <= (self.epsilon if not is_evaluate else self.args.exploration_epsilon_evaluation):
            return random.randint(0, self.num_actions - 1), None
        else:
            action, qs = self.network.q([self.phi])
            return action[0], qs[0]

    def after_action(self, state, reward, action, terminal, is_evaluate):
        self.phi[:-1] = self.phi[1:]
        self.phi[-1] = state

        if not is_evaluate:
            self.memory.add(state, reward, action, terminal)

            if self.memory.can_sample() and self.iterations > self.args.iterations_before_training and self.iterations % self.args.train_frequency == 0:
                self.epsilon = max(self.args.exploration_epsilon_end, 1 - self.network.training_iterations / self.args.exploration_epsilon_decay)
                try:
                    self.training_queue.put(self.memory.sample(), timeout=1)
                except Queue.Full:
                    pass

    def train(self):
        while True:
            try:
                states, actions, rewards, next_states, terminals, idx = self.training_queue.get(timeout=1)
                tderror, loss = self.network.train(states, actions, terminals, next_states, rewards)
            except Queue.Empty:
                # Catching an empty exception feels safer than using no timeout
                # but I'm not sure if it's strictly necessary
                pass