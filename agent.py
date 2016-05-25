from __future__ import division
import random
import numpy as np
from memory import Memory


class Agent:
    def __init__(self, args, environment, network):
        self.args = args
        self.network = network
        self.environment = environment
        self.num_actions = environment.get_num_actions()
        self.memory = Memory(args, environment)
        self.epsilon = 1
        self.iterations = 0

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

            if self.iterations > self.args.iterations_before_training and self.iterations % self.args.train_frequency == 0:
                self.train()

    def train(self):

        states, actions, rewards, next_states, terminals = self.memory.sample()
        tderror, loss = self.network.train(states, actions, terminals, next_states, rewards)

        # ideal_states, ideal_actions, ideal_rewards, ideal_next_states, ideal_terminals = self.environment.generate_test()
        #
        # self.memory = []
        # priorities = []
        # for i, s in enumerate(ideal_states):
        #     self.memory.append([ideal_states[i][0], ideal_rewards[i], ideal_actions[i], ideal_terminals[i],
        #                         ideal_next_states[i][0]])
        #     priorities.append(100.0)
        #
        # priorities = np.array(priorities)
        #
        # print self.network.q([state for i, state in enumerate(ideal_states) if ideal_actions[i] == 0])[1]
        #
        # non_terminal_trains = 0
        # non_terminal_trains_per_n = [0]
        # tmp_iterations = 3000
        # for tmp in range(tmp_iterations):
        #     states = []
        #     next_states = []
        #     actions = []
        #     terminals = []
        #     rewards = []
        #     indicies = []
        #
        #     for _ in range(self.args.batch_size):
        #         i = np.random.choice(len(self.memory), p=priorities / np.sum(priorities))
        #         experience = self.memory[i]
        #         #next_experience = self.memory[i + 1]
        #
        #         states.append([experience[0]])
        #         rewards.append(experience[1])
        #         actions.append(experience[2])
        #         terminals.append(experience[3])
        #         next_states.append([experience[4]])
        #         indicies.append(i)
        #
        #         if not terminals[-1]:
        #             non_terminal_trains += 1
        #
        #     if tmp % 50 == 0:
        #         non_terminal_trains_per_n.append(non_terminal_trains - non_terminal_trains_per_n[-1])
        #
        #     tderror, total_loss = self.network.train(ideal_states, ideal_actions, ideal_terminals, ideal_next_states, ideal_rewards)
        #
        #     policy = self.network.q([state for i, state in enumerate(ideal_states) if ideal_actions[i] == 0])[0]
        #
        #     print "Iteration:{:>5} Loss:{:>10.5}      Policy:{}".format(tmp, total_loss, "".join(str(p) if i != self.environment.goal else '-' for i, p in enumerate(policy)))
        #
        #     self.monitor.visualize_qs()
        #
        #     # for i, error in enumerate(tderror):
        #     #     priorities[i] = error ** 2