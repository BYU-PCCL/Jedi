from __future__ import division
from collections import deque
import random
import numpy as np


class Agent():
    def __init__(self, args, environment, network):
        self.args = args
        self.network = network
        self.environment = environment
        self.num_actions = environment.get_num_actions()
        self.memory = deque(maxlen=args.replay_memory_size)
        self.epsilon = 1

    def get_action(self, state, is_evaluate):
        if random.random() < (self.epsilon if not is_evaluate else self.args.exploration_epsilon_evaluation):
           return random.randint(0, self.num_actions), None
        else:
            action, qs = self.network.q(np.atleast_2d(state))
            return action[0], qs[0]

    def after_action(self, state, reward, action, terminal, is_evaluate):

        if not is_evaluate:
            self.memory.append([state, reward, action, terminal])
            self.epsilon = max(self.args.exploration_epsilon_end, 1 - self.network.training_iterations / self.args.exploration_epsilon_decay)

            if len(self.memory) > self.args.iterations_before_training:
                self.train()

    def train(self):

        ideal_states, ideal_actions, ideal_rewards, ideal_next_states, ideal_terminals = self.environment.generate_test()

        self.memory = []
        priorities = []
        for i, s in enumerate(ideal_states):
            self.memory.append([ideal_states[i][0], ideal_rewards[i], ideal_actions[i], ideal_terminals[i],
                                ideal_next_states[i][0]])
            priorities.append(100.0)

        priorities = np.array(priorities)

        non_terminal_trains = 0
        non_terminal_trains_per_n = [0]
        tmp_iterations = 3000
        for tmp in range(tmp_iterations):
            states = []
            next_states = []
            actions = []
            terminals = []
            rewards = []
            indicies = []

            for _ in range(self.args.batch_size):
                i = np.random.choice(len(self.memory), p=priorities / np.sum(priorities))
                experience = self.memory[i]
                #next_experience = self.memory[i + 1]

                states.append([experience[0]])
                rewards.append(experience[1])
                actions.append(experience[2])
                terminals.append(experience[3])
                next_states.append([experience[4]])
                indicies.append(i)

                if not terminals[-1]:
                    non_terminal_trains += 1

            if tmp % 50 == 0:
                non_terminal_trains_per_n.append(non_terminal_trains - non_terminal_trains_per_n[-1])

            tderror, total_loss = self.network.train(ideal_states, ideal_actions, ideal_terminals, ideal_next_states, ideal_rewards)

            policy = self.network.q(ideal_states)[0]

            print "Iteration:{:>5} Loss:{:>10.5}      Policy:{}".format(tmp, total_loss, "".join(str(p) if i != self.environment.goal else '-' for i, p in enumerate(policy)))

            self.monitor.visualize()

            # for i, error in enumerate(tderror):
            #     priorities[i] = error ** 2

        print "{}/{} = {:5.5}% non terminal trains of total".format(non_terminal_trains, self.args.batch_size * 1000, 100 * non_terminal_trains / (self.args.batch_size * tmp_iterations))

        for p in non_terminal_trains_per_n:
            print p / non_terminal_trains

        for i, s in enumerate(ideal_states):
            qs = self.network.q(np.atleast_2d(ideal_states[i]))[1][0]
            action = self.network.q(np.atleast_2d(ideal_states[i]))[0][0]
            error = priorities[i]
            print "{:>5} {:>5} {:>5} {:>5} {:>5} Action: {:>5} {} \t {:<+10.5}".format(ideal_states[i], ideal_actions[i], ideal_next_states[i], ideal_rewards[i], ideal_terminals[i], action, qs, error)
        quit()