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

        states = []
        next_states = []
        actions = []
        terminals = []
        rewards = []

        # for _ in range(self.args.batch_size):
        #     i = random.randint(0, len(self.memory) - 2)
        #     experience = self.memory[i]
        #     next_experience = self.memory[i + 1]
        #     states.append([experience[0]])
        #     rewards.append(experience[1])
        #     actions.append(experience[2])
        #     terminals.append(experience[3])
        #     next_states.append([next_experience[0]])

        for state in range(self.environment.size):
            for action in range(2):
                states.append([state])
                rewards.append(self.environment.reward(state, action))
                next_states.append([self.environment.transition(state, action)])
                actions.append(action)
                terminals.append(int(next_states[-1][0] == self.environment.goal))

                print states[-1], actions[-1], next_states[-1], rewards[-1], terminals[-1],  self.environment.goal

        for _ in range(10000):
            self.network.train(states, actions, terminals, next_states, rewards)
            print self.network.q(np.atleast_2d(49)), self.network.q(np.atleast_2d(51))

        quit()