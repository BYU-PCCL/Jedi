import random
import numpy as np

class ArrayEnvironment():
    def __init__ (self, args):
        self.size = 20
        self.goal = 3 #  self.size // 2
        self.position = 0
        self.episodes = 0
        self.score = 0
        self.reset()

    def get_num_actions(self):
        return 2

    def get_state_space(self):
        return [1]

    def act(self, action):
        reward = self.reward(self.position, action)

        self.position = self.transition(self.position, action)

        self.score += reward

        return self.get_state(), reward, self.get_terminal()

    def reward(self, state, action):
        next_state = self.transition(state, action)
        if next_state == self.goal:
            return 50
        return 0 #abs(self.goal - state) - abs(self.goal - next_state)

    def transition(self, state, action):
        if action == 1:
            return (state + 1) % self.size
            # return min(state + 1, self.size - 1)
        else:
            return (state - 1) % self.size
            # return max(0, (state - 1))

    def get_episodes(self):
        return self.episodes

    def get_score(self):
        return self.score

    def get_state(self):
        return np.array(self.position)

    def max_state_value(self):
        return self.size

    def get_terminal(self):
        return self.position == self.goal

    def reset(self):
        self.episodes += 1
        self.score = 0
        self.position = random.randint(0, self.size - 1)

    def generate_test(self):
        states = []
        next_states = []
        actions = []
        terminals = []
        rewards = []

        for state in range(self.size):
            for action in range(2):
                states.append([state])
                rewards.append(self.reward(state, action))
                next_states.append([self.transition(state, action)])
                actions.append(action)
                terminals.append(int(next_states[-1][0] == self.goal))

        return states, actions, rewards, next_states, terminals

