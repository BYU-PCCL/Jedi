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
        self.ready_queue = Queue.Queue(maxsize=self.args.threads)

        self.phi = np.zeros(tuple([args.phi_frames]) + environment.get_state_space(), dtype=np.uint8)

        self.threads = []
        for id in range(args.threads):
            self.threads.append(Thread(target=self.train, args=[id]))
            self.threads[-1].setDaemon(True)

    def get_action(self, state, is_evaluate):
        self.iterations += 1
        if random.random() <= (self.epsilon if not is_evaluate else self.args.exploration_epsilon_evaluation):
            return random.randint(0, self.num_actions - 1), None
        else:
            action, qs, _ = self.network.q(states=[self.phi])
            return action[0], qs[0]

    def after_action(self, state, reward, action, terminal, is_evaluate):
        self.phi[:-1] = self.phi[1:]
        self.phi[-1] = state

        self.memory.add(state, reward, action, terminal)

        if self.iterations > self.args.iterations_before_training and self.iterations % self.args.train_frequency == 0:
            if not self.threads[0].isAlive():
                for thread in self.threads:
                    thread.start()

            try:
                self.ready_queue.put(True, timeout=1)  # Wait for training to finish
                self.epsilon = max(self.args.exploration_epsilon_end, 1 - self.network.training_iterations / self.args.exploration_epsilon_decay)
            except Queue.Full:
                pass

    def train(self, id):
        while True:
            self.ready_queue.get()  # Notify main thread a training has complete
            states, actions, rewards, next_states, terminals, lookaheads, idx = self.memory.sample()
            tderror, loss = self.network.train(states=states, actions=actions, terminals=terminals, next_states=next_states, rewards=rewards)
            self.memory.update(idx, priority=tderror)

import time
class Test(Agent):
    def __init__(self, args, environment, network):
        Agent.__init__(self, args, environment, network)

        self.policy_test = environment.generate_test()
        ideal_states, ideal_actions, ideal_rewards, ideal_next_states, ideal_terminals = self.policy_test

        for _ in range(5000):
            error, loss = self.network.train(states=ideal_states, actions=ideal_actions, terminals=ideal_terminals, next_states=ideal_next_states, rewards=ideal_rewards)

            policy, qs, _ = self.network.q(states=[[[s] for _ in range(self.args.phi_frames)] for s in range(self.environment.size)])
            _, goal_q, _ = self.network.q(states=[[[self.environment.goal - 1] for _ in range(self.args.phi_frames)]])
            print self.network.training_iterations, \
                "".join(str(p) if i != self.environment.goal else '-' for i, p in enumerate(policy)), \
                np.max(goal_q), \
                loss

        for s in range(self.environment.size):
            print s if s != self.environment.goal else '-', list(self.network.q(states=[[[s] for _ in range(self.args.phi_frames)]])[1][0])


class Convergence(Agent):
    def __init__(self, args, environment, network):
        Agent.__init__(self, args, environment, network)
        self.convergence_repetitions = args.convergence_repetitions

    def train(self, id):
        while True:
            try:
                self.ready_queue.get(timeout=1)  # Notify main thread a training has complete

                for _ in range(self.convergence_repetitions):
                    states, actions, rewards, next_states, terminals, lookaheads, idx = self.memory.sample()
                    priority, loss = self.network.train(states=states,
                                                        actions=actions,
                                                        terminals=terminals,
                                                        next_states=next_states,
                                                        rewards=rewards)
                    self.memory.update(idx, priority=priority)

            except Queue.Empty:
                pass


class QExplorer(Agent):
    def __init__(self, args, environment, network):
        Agent.__init__(self, args, environment, network)

    def get_action(self, state, is_evaluate):
        self.iterations += 1

        if self.iterations < self.args.iterations_before_training:
            return random.randint(0, self.num_actions - 1), None

        action, qs, _ = self.network.q(states=[self.phi])
        qprob = qs[0] - np.min(qs[0]) + 0.01
        qprob += np.mean(qprob) * self.epsilon  # ensures everyone has chance of being selected, but decays over time
        qprob = qprob / np.sum(qprob)

        return np.random.choice(self.num_actions, p=qprob), qs[0]


class DensityExplorer(Agent):
    def __init__(self, args, environment, network):
        Agent.__init__(self, args, environment, network)

    def get_action(self, state, is_evaluate):
        self.iterations += 1

        if self.iterations < self.args.iterations_before_training:
            return random.randint(0, self.num_actions - 1), None

        action, qs, additional_ops = self.network.q(states=[self.phi])
        variances = additional_ops[0]

        if random.random() <= (self.epsilon if not is_evaluate else self.args.exploration_epsilon_evaluation):
            vprob = variances[0] - np.min(variances[0]) + 0.01
            vprob = vprob / np.sum(vprob)

            return np.random.choice(self.num_actions, p=vprob), qs[0]

        else:
            return action[0], qs[0]