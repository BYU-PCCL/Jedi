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
        self.training_queue = Queue.Queue(maxsize=1)

        self.train_thread = Thread(target=self.train)
        self.train_thread.setDaemon(True)
        self.train_thread.start()

    def get_action(self, state, is_evaluate):
        self.iterations += 1
        if random.random() < (self.epsilon if not is_evaluate else self.args.exploration_epsilon_evaluation):
            return random.randint(0, self.num_actions - 1), None
        else:
            action, qs = self.network.q([self.memory.get_recent()])
            return action[0], qs[0]

    def after_action(self, state, reward, action, terminal, is_evaluate):
        if not is_evaluate:
            self.memory.add(state, reward, action, terminal)
            self.epsilon = max(self.args.exploration_epsilon_end, 1 - self.network.training_iterations / self.args.exploration_epsilon_decay)

            if self.memory.can_sample() and self.iterations > self.args.iterations_before_training and self.iterations % self.args.train_frequency == 0:
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

        # for i, _ in enumerate(states):
        #     print states[i], actions[i], next_states[i], rewards[i], terminals[i]

        # ideal_states, ideal_actions, ideal_rewards, ideal_next_states, ideal_terminals = self.environment.generate_test()
        #
        # self.memory = Memory(self.args, self.environment)
        # priorities = []
        # for i, s in enumerate(ideal_states):
        #     print ideal_states[i], ideal_actions[i], ideal_rewards[i], ideal_next_states[i], ideal_terminals[i]
        #     self.memory.add(np.array(ideal_states[i]), ideal_rewards[i], ideal_actions[i], ideal_terminals[i])
        #     priorities.append(100.0)
        #
        # priorities = np.array(priorities)
        #
        # for tmp in range(3000):
        #     states, actions, rewards, next_states, terminals = self.memory.sample()
        #     next_states = np.array([self.environment.transition(state, actions[i]) for i, state in enumerate(states)])
        #
        #     tderror, total_loss = self.network.train(states, actions, terminals, next_states, rewards)
        #
        #     policy = self.network.q([[state] for i, state in enumerate(ideal_states) if ideal_actions[i] == 0])[0]
        #
        #     print "Iteration:{:>5} Loss:{:>10.5}      Policy:{}".format(tmp, total_loss, "".join(str(p) if i != self.environment.goal else '-' for i, p in enumerate(policy)))
        #
        #     #self.monitor.visualize_qs()
        #
        #     # for i, error in enumerate(tderror):
        #     #     priorities[i] = error ** 2