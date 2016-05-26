import random
import numpy as np
import gym
import cv2

class ArrayEnvironment:
    def __init__(self, args):
        self.size = 20
        self.goal = 4 #self.size // 2
        self.position = [0]
        self.episodes = 0
        self.score = 0
        self.reset()

    def get_num_actions(self):
        return 2

    def get_state_space(self):
        return tuple([1])

    def act(self, action):

        reward = self.reward(self.position, action)

        self.position = self.transition(self.position, action)

        self.score += reward
        self.frames += 1

        return self.get_state(), reward, self.get_terminal()

    def reward(self, state, action):
        next_state = self.transition(state, action)

        if next_state[0] == self.goal:
            return 50
        return 0 #abs(self.goal - state) - abs(self.goal - next_state)

    def transition(self, state, action):
        if action == 1:
            #return [(state[0] + 1) % self.size]
            return [min(state[0] + 1, self.size - 1)]
        else:
            #return [(state[0] - 1) % self.size]
            return [max(0, (state[0] - 1))]

    def get_episodes(self):
        return self.episodes

    def get_score(self):
        return self.score

    def get_state(self):
        return np.array(self.position)

    def max_state_value(self):
        return self.size

    def get_terminal(self):
        return self.position[0] == self.goal

    def reset(self):
        self.episodes += 1
        self.score = 0
        self.frames = 0

        while True:
            self.position = [random.randint(0, self.size - 1)]
            if abs(self.position[0] - self.goal) > 3:
                break

    def generate_test(self):
        states = []
        next_states = []
        actions = []
        terminals = []
        rewards = []

        for state in range(self.size):
            for action in range(2):
                states.append([state])
                rewards.append(self.reward([state], action))
                next_states.append([self.transition([state], action)])
                actions.append(action)

                terminals.append(next_states[-1][0][0] == self.goal)

        return states, actions, rewards, next_states, terminals


class AtariEnvironment:
    def __init__(self, args):
        self.args = args
        self.env = gym.make('Breakout-gray-v0')

        self.score = 0
        self.episodes = 0
        self.terminal = False
        self.frames = 0
        self.state = np.zeros((args.resize_width, args.resize_height), dtype=np.uint8)
        self.buffer = np.zeros((args.buffer_size, args.resize_height, args.resize_width), dtype=np.uint8)
        self.reset()

    def get_num_actions(self):
        return self.env.action_space.n

    def get_state_space(self):
        return self.state.shape

    def act(self, action):
        total_reward = 0
        self.frames += 1
        for _ in range(self.args.actions_per_tick):
            screen, reward, self.terminal, _ = self.env.step(action)
            total_reward += reward

            if self.terminal:
                break

        self.state = cv2.resize(screen, (self.args.resize_width, self.args.resize_height))

        # Roll the buffer
        # Add a resized, grayscale image to the buffer
        # self.buffer[1:, ...] = self.buffer[0:-1, ...]
        # self.buffer[-1, ...] = frame

        # self.state = np.max(self.buffer, axis=0)

        self.score += total_reward

        return self.get_state(), reward, self.terminal

    def get_episodes(self):
        return self.episodes

    def get_score(self):
        return self.score

    def get_state(self):
        return self.state

    def max_state_value(self):
        return np.max(self.env.observation_space.high)

    def min_state_value(self):
        return np.min(self.env.observation_space.low)

    def reset(self):
        self.episodes += 1
        self.frames = 0
        self.env.reset()

        self.buffer.fill(0)
        self.state.fill(0)
        self.terminal = False
        self.score = 0

        if self.args.max_initial_noop > 0:
            for _ in range(random.randint(0, self.args.max_initial_noop)):
                self.act(0)

    def generate_test(self):
        return None
