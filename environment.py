import random
import numpy as np

class ArrayEnvironment():
    def __init__ (self, args):
        self.size = 20
        self.goal = self.size // 2
        self.position = 0
        self.episodes = 0
        self.score = 0
        self.reset()

    def get_num_actions(self):
        return 2

    def get_state_space(self):
        return [1]

    def dist_to_goal(self):
        return abs(self.goal - self.position)

    def act(self, action):
        before = self.dist_to_goal()
        self.position = self.transition(self.position, action)

        after = self.dist_to_goal()

        reward = before - after
        self.score += reward

        return self.get_state(), reward, self.get_terminal()

    def reward(self, state, action):
        next_state = self.transition(state, action)
        if next_state == self.goal:
            return 50
        return 0 #abs(self.goal - state) - abs(self.goal - next_state)

    def transition(self, state, action):
        if action == 0:
            return min(state + 1, self.size - 1)
        else:
            return max(0, (state - 1))

    def get_episodes(self):
        return self.episodes

    def get_score(self):
        return self.score

    def get_state(self):
        return np.array(self.position)

    def get_terminal(self):
        return self.position == self.size // 2

    def reset(self):
        self.episodes += 1
        self.score = 0
        self.position = random.randint(0, self.size - 1)
