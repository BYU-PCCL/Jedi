import random
import numpy as np
import gym
import tensorflow as tf
from enum import Enum

class ArrayEnvironment:
    def __init__(self, args):
        self.args = args
        self.size = 20
        self.goal = 4  # self.size // 2
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
        return 0  # abs(self.goal - state) - abs(self.goal - next_state)

    def transition(self, state, action):
        if action == 1:
            return [(state[0] + 1) % self.size]
            #return [min(state[0] + 1, self.size - 1)]
        else:
            return [(state[0] - 1) % self.size]
            #return [max(0, (state[0] - 1))]

    def get_episodes(self):
        return self.episodes

    def get_score(self):
        return self.score

    def get_state(self):
        return np.array(self.position)

    def max_state_value(self):
        return self.size

    def get_terminal(self):
        return self.position[0] == self.goal or self.frames >= self.args.max_frames_per_episode

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
                states.append([[state] for _ in range(self.args.phi_frames)])
                rewards.append(self.reward([state], action))
                next_states.append([self.transition([state], action) for _ in range(self.args.phi_frames)])
                actions.append(action)

                terminals.append(next_states[-1][0][0] == self.goal)

        return states, actions, rewards, next_states, terminals


class AtariEnvironment:
    def __init__(self, args):
        self.args = args
        self.env = gym.make(args.rom + '-gray-v0')

        self.score = 0
        self.episodes = 0
        self.terminal = False
        self.frames = 0
        self.lives = 0
        self.state = np.zeros((args.resize_width, args.resize_height), dtype=np.uint8)
        self.buffer = np.zeros((args.buffer_size, args.resize_height, args.resize_width), dtype=np.uint8)

        height, width, channels = self.env.observation_space.shape
        self.resize_input = tf.placeholder(np.uint8, shape=[None, height, width, channels])
        self.resize_op = tf.image.resize_bilinear(self.resize_input, [args.resize_height, args.resize_width])

        tf.set_random_seed(args.random_seed)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction,
                                    allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                                                     allow_soft_placement=True,
                                                     log_device_placement=self.args.verbose))

        self.reset()

    def get_num_actions(self):
        return self.env.action_space.n

    def get_state_space(self):
        return self.state.shape

    def act(self, action):
        total_reward = 0
        self.frames += 1
        reward = 0
        for _ in range(self.args.actions_per_tick):
            screen, reward, self.terminal, _ = self.env.step(action)
            total_reward += reward

            if self.terminal:
                break

        self.terminal = self.terminal or self.frames >= self.args.max_frames_per_episode

        with self.sess.as_default():
            frame = self.resize_op.eval(feed_dict={self.resize_input: [screen]})[0, :, :, 0]

        if self.lives > self.env.ale.lives() and self.args.negative_reward_on_death:
            total_reward -= 10

        self.lives = self.env.ale.lives()

        # Roll the buffer
        # Add a resized, grayscale image to the buffer
        if self.args.buffer_size > 1:
            self.buffer[1:, ...] = self.buffer[0:-1, ...]
            self.buffer[-1, ...] = frame
            self.state = np.max(self.buffer, axis=0)
        else:
            self.state = frame

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
            for _ in range(random.randint(0, self.args.max_initial_noop // self.args.actions_per_tick)):
                self.act(0)

    def generate_test(self):
        return None


class MazeEnvironment:
    class Objects(Enum):
        user = 0
        wall = 1
        open = 4
        coin = 7
        goal = 10

    class States(Enum):
        user = 75
        wall = 125
        open = 0
        goal = 200

    def __init__(self, args):
        self.args = args
        self.episodes = 0

        self.maze, self.start = self._build_maze()

        self.rewards = {
            self.Objects.goal: 10.0,
            self.Objects.coin: 1.0,
            self.Objects.wall: 0.0,
            self.Objects.open: 0.0
        }

        self.reset()

    def get_num_actions(self):
        return 4

    def get_state_space(self):
        return self.get_state().shape

    def act(self, action):
        self.frames += 1

        delta_position = np.array({
            0: (-1, 0),  # Down
            1: (1, 0),   # Up
            2: (0, 1),   # Right
            3: (0, -1)   # Left
        }[action])

        proposed_position = self.position + delta_position

        reward = self.rewards[self.get_maze_position(proposed_position)]

        if self.get_maze_position(proposed_position) != self.Objects.wall:
            self.position = proposed_position

        self.score += reward

        return self.get_state(self.position), reward, self.get_terminal()

    def transition(self, state, action):
        delta_position = np.array({
              0: (-1, 0),  # Down
              1: (1, 0),  # Up
              2: (0, 1),  # Right
              3: (0, -1)  # Left
          }[action])

        position = np.unravel_index(state, self.maze.shape)
        proposed_position = position + delta_position

        if self.get_maze_position(proposed_position) != self.Objects.wall:
            return self.get_state(proposed_position)

        return state

    def get_maze_position(self, position):
        x, y = position
        max_x, max_y = self.maze.shape

        return self.maze[np.clip(x, 0, max_x), np.clip(y, 0, max_y)]

    def get_episodes(self):
        return self.episodes

    def get_score(self):
        return self.score

    def get_maze_without_agent(self):
        goals = np.array(self.maze == self.Objects.goal, dtype=np.uint8) * self.States.goal
        state = np.array(self.maze == self.Objects.wall, dtype=np.uint8) * self.States.wall
        return state + goals

    def get_state(self, position):
        state = np.zeros(np.prod(self.maze.shape))
        state[np.ravel_multi_index(position, self.maze.shape)] = 1
        return np.atleast_2d(state)

    def max_state_value(self):
        return 1

    def get_terminal(self):
        self.terminal = self.terminal or ((self.get_maze_position(self.position) == self.Objects.goal) or self.frames >= self.args.max_frames_per_episode)
        return self.terminal

    def _build_maze(self):
        l, _, G = self.Objects.wall, self.Objects.open, self.Objects.goal
        start = np.array([1, 1])
        return np.array([
            [l, l, l, l, l, l, l, l, l, l, l, l],  # [w w w w w w w w w w w w]
            [l, _, _, _, _, _, _, _, _, _, G, l],  # [w       w       w     w]
            [l, _, _, l, G, l, _, _, _, _, _, l],  # [w   w       w       w w]
            [l, _, _, _, l, _, _, _, l, _, _, l],  # [w       w       w     w]
            [l, _, l, _, _, _, l, _, _, _, l, l],  # [w   w       w       w w]
            [l, _, _, _, l, _, _, _, l, _, _, l],  # [w       w       w     w]
            [l, _, l, _, _, _, l, _, _, _, l, l],  # [w   w       w       w w]
            [l, _, G, _, l, _, _, _, l, _, _, l],  # [w       w       w     w]
            [l, _, l, _, _, _, l, _, _, _, l, l],  # [w   w       w       w w]
            [l, _, _, _, l, _, _, _, l, _, _, l],  # [w       w       w     w]
            [l, _, l, _, _, _, l, _, _, _, G, l],  # [w   w       w       w w]
            [l, l, l, l, l, l, l, l, l, l, l, l]   # [w w w w w w w w w w w w]
        ]), start

    def reset(self):
        self.episodes += 1
        self.score = 0
        self.frames = 0
        self.terminal = False

        self.position = self.start