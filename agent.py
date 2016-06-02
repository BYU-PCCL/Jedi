from __future__ import division
import random
import numpy as np
from memory import Memory
import Queue
from threading import Thread


class Agent(object):
    def __init__(self, args, environment, network):
        self.args = args
        self.network = network
        self.environment = environment
        self.num_actions = environment.get_num_actions()
        self.memory = Memory(args, environment)
        self.epsilon = 1
        self.iterations = 0
        self.training_queue = Queue.Queue(maxsize=args.threads)
        self.ready_queue = Queue.Queue(maxsize=args.threads)

        self.phi = np.zeros([args.phi_frames, args.resize_height, args.resize_width], dtype=np.uint8)

        self.threads = []
        for _ in range(args.threads):
            self.threads.append(Thread(target=self.train))
            self.threads[-1].setDaemon(True)
            self.threads[-1].start()

        self.sample_thread = Thread(target=self.generate_samples)
        self.sample_thread.setDaemon(True)
        self.sample_thread.start()

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

            if self.iterations > self.args.iterations_before_training and self.iterations % self.args.train_frequency == 0:
                self.ready_queue.put(True)  # Wait for training to finish
                self.epsilon = max(self.args.exploration_epsilon_end, 1 - self.network.training_iterations / self.args.exploration_epsilon_decay)

    def generate_samples(self):
        while True:
            if self.memory.can_sample():
                self.training_queue.put(self.memory.sample())

    def train(self):
        while True:
            self.ready_queue.get()  # Notify main thread a training has complete
            states, actions, rewards, next_states, terminals, lookaheads, idx = self.training_queue.get()
            tderror, loss = self.network.train(states, actions, terminals, next_states, rewards, lookaheads)
            self.memory.update(idx, np.abs(tderror))

class QExplorer(Agent):
    def __init__(self, args, environment, network):
        Agent.__init__(self, args, environment, network)

    def get_action(self, state, is_evaluate):
        self.iterations += 1

        if self.iterations < self.args.iterations_before_training:
            return random.randint(0, self.num_actions - 1), None

        action, qs = self.network.q([self.phi])
        qprob = qs[0] - np.min(qs[0]) + 0.01
        qprob += np.mean(qprob) * self.epsilon  # ensures everyone has chance of being selected, but decays over time
        qprob = qprob / np.sum(qprob)

        return np.random.choice(self.num_actions, p=qprob), qs[0]


class DensityExplorer(Agent):
    def __init__(self, args, environment, network):
        Agent.__init__(self, args, environment, network)
        assert args.network_type == 'density', 'Density Explorer must use the Density Network'

    def get_action(self, state, is_evaluate):
        self.iterations += 1

        if self.iterations < self.args.iterations_before_training:
            return random.randint(0, self.num_actions - 1), None

        action, qs, variances = self.network.q([self.phi])

        if random.random() <= (self.epsilon if not is_evaluate else self.args.exploration_epsilon_evaluation):
            vprob = variances[0] + 1.0
            vprob = vprob / np.sum(vprob)

            if self.iterations % 1000 == 0:
                print vprob, variances[0], " "

            return np.random.choice(self.num_actions, p=vprob), qs[0]

        else:
            return action[0], qs[0]