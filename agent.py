from __future__ import division
import random
import numpy as np
from memory import Memory
import sys
import scipy.sparse.linalg as scipy

if sys.version[0] == '2':
    import Queue as Queue
else:
    import queue as Queue
from threading import Thread


class Agent(object):
    def __init__(self, args, environment, network):
        self.args = args
        self.network = network
        self.environment = environment
        self.num_actions = environment.get_num_actions()
        self.memory = Memory(args, environment)
        self.epsilon = .99999
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

    def start_threads(self):
        if not self.threads[0].isAlive():
            for thread in self.threads:
                thread.start()

    def after_action(self, state, reward, action, terminal, is_evaluate):
        self.phi[:-1] = self.phi[1:]
        self.phi[-1] = state

        self.memory.add(state, reward, action, terminal)

        if self.iterations > self.args.iterations_before_training and self.iterations % self.args.train_frequency == 0:
            self.start_threads()

            try:
                self.ready_queue.put(True, timeout=1)  # Wait for training to finish
                self.epsilon = max(self.args.exploration_epsilon_end,
                                   1 - self.network.training_iterations / self.args.exploration_epsilon_decay)
            except Queue.Full:
                pass

    def train(self, id):
        while True:
            self.ready_queue.get()  # Notify main thread a training has complete
            states, actions, rewards, next_states, terminals, lookaheads, idx = self.memory.sample()
            tderror, loss = self.network.train(states=states, actions=actions, terminals=terminals,
                                               next_states=next_states, rewards=rewards)
            self.memory.update(idx, priority=tderror)


class Test(Agent):
    def __init__(self, args, environment, network):
        Agent.__init__(self, args, environment, network)

        self.policy_test = environment.generate_test()
        ideal_states, ideal_actions, ideal_rewards, ideal_next_states, ideal_terminals = self.policy_test

        for _ in range(3000):
            error, loss = self.network.train(states=ideal_states,
                                             actions=ideal_actions,
                                             terminals=ideal_terminals,
                                             next_states=ideal_next_states,
                                             rewards=ideal_rewards)

            # print loss

            policy, qs, _ = self.network.q(
                states=[[[s] for _ in range(self.args.phi_frames)] for s in range(self.environment.size)])

            _, goal_q, _ = self.network.q(states=[[[self.environment.goal - 1] for _ in range(self.args.phi_frames)]])

            print(self.network.training_iterations,
                  "".join(str(p) if i != self.environment.goal else '-' for i, p in enumerate(policy)),
                  np.max(goal_q),
                  loss)

        for s in range(self.environment.size):
            print(s if s != self.environment.goal else '-',
                  list(self.network.q(states=[[[s] for _ in range(self.args.phi_frames)]])[1][0]))


class Lookahead(Agent):
    def __init__(self, args, environment, network):
        Agent.__init__(self, args, environment, network)

    def train(self, id):
        while True:
            self.ready_queue.get()  # Notify main thread a training has complete
            states, actions, rewards, next_states, terminals, lookaheads, idx = self.memory.sample()
            tderror, loss = self.network.train(states=states, actions=actions, terminals=terminals,
                                               next_states=next_states, rewards=rewards, lookaheads=lookaheads)
            self.memory.update(idx, priority=tderror)


class Convergence(Agent):
    def __init__(self, args, environment, network):
        Agent.__init__(self, args, environment, network)

        self.sample_queue = Queue.Queue(maxsize=self.args.threads * args.convergence_repetitions)

        del self.threads[1:]  # Only one training thread allowed

        self.sample_threads = []
        for id in range(args.convergence_sample_threads):
            self.sample_threads.append(Thread(target=self.sample, args=[id]))
            self.sample_threads[-1].setDaemon(True)

    def start_threads(self):
        if not self.threads[0].isAlive():
            for thread in self.threads:
                thread.start()

            for sample_thread in self.sample_threads:
                sample_thread.start()

    def sample(self, id):
        while True:
            try:
                self.sample_queue.put(self.memory.sample(), timeout=.20)
            except Queue.Full:
                pass

    def train(self, id):
        while True:
            try:
                self.ready_queue.get(timeout=1)  # Notify main thread a training has complete

                for _ in range(self.args.convergence_repetitions):
                    states, actions, rewards, next_states, terminals, lookaheads, idx = self.sample_queue.get(timeout=1)
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

        if not is_evaluate:
            qprob = qs[0] - np.min(qs[0]) + 0.01
            qprob = qprob ** 2.5
            qprob = qprob / np.sum(qprob)

            return np.random.choice(self.num_actions, p=qprob), qs[0]
        else:
            return action[0], qs[0]


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


class ExperienceAsAModel(Agent):
    def __init__(self, args, environment, network):
        Agent.__init__(self, args, environment, network)

        self.action_probabilities = np.zeros(args.replay_memory_size, dtype=np.float32)
        self.state_probabilities = np.zeros(args.replay_memory_size, dtype=np.float32)
        self.last_action_probability = 0

        self.connectivity_matrix = self.environment.get_connectivity_matrix()

    def get_action(self, state, is_evaluate):

        action, qs = Agent.get_action(self, state, is_evaluate)

        # Probability of an epsilon-greedy random exploration action
        # p = (epsilon / N)
        self.last_action_probability = self.epsilon / self.environment.get_num_actions()

        # Probability of an epsilon-greedy policy action
        # p = (epsilon / N) + (1 - epsilon)
        if qs is not None:
            self.last_action_probability += (1 - self.epsilon)

        return action, qs

    def after_action(self, state, reward, action, terminal, is_evaluate):

        # MUST be done before self.memory.add
        self.action_probabilities[self.memory.current] = self.last_action_probability
        self.state_probabilities[self.memory.current] = 1  # self.state_distribution_given_policy()[state.argmax()]

        return Agent.after_action(self, state, reward, action, terminal, is_evaluate)

    def state_distribution_given_policy(self):
        states = np.expand_dims(np.eye(144), 1)
        states = np.expand_dims(states, 1)
        actions, qs, _ = self.network.q(states=states)

        states = np.linspace(0, 144, num=144, endpoint=False, dtype=np.int64)

        epsilon = self.args.exploration_epsilon_evaluation
        matrix = self.connectivity_matrix * (epsilon / self.environment.get_num_actions())

        for i, s in enumerate(states):
            if self.environment.get_object_at_state(s) != self.environment.Objects.wall:
                matrix[s, self.environment.transition_indexed(s, actions[i])] += (1 - epsilon)

        agent_states = matrix.sum(0) > 0
        matrix = matrix[agent_states][:, agent_states]

        # initial_vector = np.random.random(matrix.shape[0])
        # initial_vector /= initial_vector.sum()
        # w, v = scipy.eigs(matrix.T)
        #
        # distribution = v[:, 0].real
        # distribution -= distribution.min()
        # distribution /= distribution.sum()

        return [1] * 144

    def distribution_action_given_state(self, is_on_policy, states):
        # p(a | s ; current_policy)
        alpha = self.epsilon / self.environment.get_num_actions()
        return ((1 - self.epsilon) * is_on_policy) + alpha

    def policy(self):
        states = np.expand_dims(np.eye(144), 1)
        states = np.expand_dims(states, 1)
        actions, qs, _ = self.network.q(states=states)

        return actions

    def train(self, id):
        while True:
            self.ready_queue.get()  # Notify main thread a training has complete
            states, actions, rewards, next_states, terminals, lookaheads, idx = self.memory.sample()
            old_policy_action_probabilites = self.action_probabilities[idx].copy()
            old_policy_state_probabilites = self.state_probabilities[idx].copy()

            policy_actions, qs, additional_ops = self.network.q(states=states)
            on_policy_mask = actions == policy_actions

            current_policy_action_probabilities = self.distribution_action_given_state(on_policy_mask, states)
            # current_policy_state_probabilities = self.distribution_state(states)

            batch_size = states.shape[0]
            weights = np.ones(batch_size) / batch_size

            weights *= np.nan_to_num(current_policy_action_probabilities / old_policy_action_probabilites)
            # weights *= np.nan_to_num(current_policy_state_probabilities / old_policy_state_probabilites)

            weights = np.clip(weights, 0, 10)

            if id == 0:
                print np.array_str(self.policy(), max_line_width=144)

            tderror, loss = self.network.train(states=states,
                                               actions=actions,
                                               terminals=terminals,
                                               next_states=next_states,
                                               rewards=rewards,
                                               weights=weights)

            self.memory.update(idx, priority=tderror)


            # todo distributed:
            # class Distributed_Runner()
            # switch thread_id % n:
            # case 0:
            # set agent as (w1, t1, a1)
