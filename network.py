import tensorflow as tf
import subprocess
import os
import numpy as np
import shutil

class TrainTarget(object):
    def __init__(self, Type, args, environment):
        self.args = args
        self.sess = Type.create_session(args)

        self.target_network = Type(args, environment, 'target', self.sess)
        self.train_network = Type(args, environment, 'train', self.sess)

        self.training_iterations = 0
        self.lr = 0.0
        self.batch_loss = 0.0

        self.copy = [
            weight.assign(args.target_network_alpha * self.train_network.weights[i] + (1.0 - args.target_network_alpha) * weight, use_locking=True)
                for i, weight in enumerate(self.target_network.weights)
        ] + [
            bias.assign(args.target_network_alpha * self.train_network.biases[i] + (1.0 - args.target_network_alpha) * bias, use_locking=True)
                for i, bias in enumerate(self.target_network.biases)
        ]

    def __len__(self):
        return len(self.train_network)

    def update(self):
        self.sess.run(self.copy)

    def q(self, states):
        return self.target_network.q(states)

    def train(self, states, actions, terminals, next_states, rewards, lookahead=None):
        response = self.train_network.train(states, actions, terminals,
                                            next_states, rewards, lookahead, self.target_network)
        self.training_iterations = self.train_network.training_iterations
        self.lr = self.train_network.lr
        self.batch_loss = self.train_network.batch_loss

        if self.training_iterations % self.args.copy_frequency == 0:
            self.update()

        return response


class Network(object):
    def __init__(self, args, environment, name='network', sess=None):
        self.args = args
        self.environment = environment
        self.name = name
        self.training_iterations = 0
        self.batch_loss = 0
        self.lr = 0
        self.default_initializer = args.initializer

        self.weights = []
        self.biases = []
        self.activations = []
        self.additional_q_ops = []

        self.sess = self.create_session(args) if sess is None else sess

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        with tf.name_scope('input') as scope:
            self.state = self.float([None, args.phi_frames] + list(environment.get_state_space()), name='state') / float(environment.max_state_value())
            self.lookahead = self.float([None, args.phi_frames] + list(environment.get_state_space()), name='lookahead') / float(environment.max_state_value())
            self.next_q = self.float([None], name='next_q')
            self.action = self.int([None], name='action_index')
            self.action_one_hot = self.one_hot(self.action, self.environment.get_num_actions(), name='one_hot_action')
            self.terminal = self.float([None], name='terminal')
            self.reward = self.float([None], name='reward')
            self.processed_reward = tf.clip_by_value(self.reward, -1.0, 1.0) if self.args.clip_reward else self.reward

        self.output = None

    def __len__(self):
        return sum([sum([reduce(lambda x, y: x * y, l.get_shape().as_list()) for l in e]) for e in [self.weights, self.biases]])

    def get_loss(self, processed_delta, prediction, truth):
        return tf.reduce_mean(tf.square(processed_delta), name='loss')

    def post_init(self):
        with tf.name_scope('output') as scope:
            self.q_action = self.argmax(self.output)
            self.q_max = self.max(self.output)

        with tf.name_scope('target_q') as scope:
            target_q = self.processed_reward + self.args.discount * (1.0 - self.terminal) * self.next_q

        with tf.name_scope('delta') as scope:
            q_acted = self.sum(self.output * self.action_one_hot, name='q_acted')
            self.delta = target_q - q_acted
            self.processed_delta = tf.clip_by_value(self.delta, -1.0, 1.0) if self.args.clip_tderror else self.delta

        with tf.name_scope('loss') as scope:
            self.loss = self.get_loss(self.processed_delta, truth=target_q, prediction=q_acted)

        with tf.name_scope('learning_rate') as scope:
            # Learning Rate Calculation
            self.learning_rate = tf.maximum(self.args.learning_rate_end,
                                            tf.train.exponential_decay(
                                              self.args.learning_rate_start,
                                              self.global_step,
                                              self.args.learning_rate_decay_step,
                                              self.args.learning_rate_decay,
                                              staircase=False))

        with tf.name_scope('optimizer') as scope:
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate,
                                                       decay=self.args.rms_decay,
                                                       momentum=float(self.args.rms_momentum),
                                                       epsilon=self.args.rms_eps).minimize(self.loss,
                                                                                           global_step=self.global_step)

        # Initialize
        self.sess.run(tf.initialize_all_variables())
        self.tensorboard()

    def tensorboard(self):
        shutil.rmtree(self.args.tf_summary_path)
        tf.train.SummaryWriter(self.args.tf_summary_path, self.sess.graph)
        subprocess.Popen(["tensorboard", "--logdir=" + self.args.tf_summary_path],
                         stdout=open(os.devnull, 'w'),
                         stderr=open(os.devnull, 'w'),
                         close_fds=True)

    @staticmethod
    def create_session(args):
        tf.set_random_seed(args.random_seed)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction, allow_growth=True)
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    def one_hot(self, source, size, name='onehot'):
        return tf.one_hot(source, size, 1.0, 0.0, name=name)

    def argmax(self, source, name='argmax'):
        return tf.argmax(source, dimension=1)

    def max(self, source, name='max'):
        return tf.reduce_max(source, reduction_indices=1, name=name)

    def flatten(self, source, name='flatten'):
        shape = source.get_shape().as_list()
        dim = reduce(lambda x, y: x*y, shape[1:])

        with tf.variable_scope(name + "_flatten"):
            return tf.reshape(source, [-1, dim], name=name)

    def expand(self, source, dim, name='expand'):
        with tf.variable_scope(name + "_expand"):
            return tf.expand_dims(source, dim)

    def merge(self, left, right, idx=1, name='merge'):
        with tf.variable_scope(name + "_merge"):
            return tf.concat(idx, [left, right])

    def sum(self, source, name, idx=1):
        return tf.reduce_sum(source, reduction_indices=idx, name=name)

    def squared_sum(self, source, name):
        return tf.reduce_sum(tf.square(source), reduction_indices=0, name=name)

    def parse_initializer(self, initializer, stddev):
        return {
            'normal': tf.random_normal_initializer(stddev=stddev),
            'xavier': tf.contrib.layers.xavier_initializer(),
            'uniform': tf.random_uniform_initializer(),
            'truncated-normal': tf.truncated_normal_initializer(0, stddev=stddev)
        }[initializer if initializer != 'default' else self.default_initializer]

    def parse_activation(self, activation):
        return {
            'relu': tf.nn.relu,
            'sigmoid': tf.nn.sigmoid,
            'none': None}[activation]

    def linear(self, source, output_size, stddev=0.02, initializer='default', bias_start=0.01, activation_fn='relu', name='linear', w=None, b=None):
        shape = source.get_shape().as_list()

        initializer = self.parse_initializer(initializer, stddev)
        activation_fn = self.parse_activation(activation_fn)

        with tf.variable_scope(name + '_linear') as scope:
            if w is None:
                w = tf.get_variable("weight", [shape[1], output_size], tf.float32, initializer)
                self.weights.append(w)
            if b is None:
                b = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))
                self.biases.append(b)

            out = tf.nn.bias_add(tf.matmul(source, w), b)
            activated = activation_fn(out) if activation_fn is not None else out
            self.activations.append(activated)

            return activated, w, b

    def conv2d(self, source, size, filters, stride, padding='SAME', stddev=0.02, initializer='default', bias_start=0.01,
               activation_fn='relu', name='conv2d', w=None, b=None):
        shape = source.get_shape().as_list()
        initializer = self.parse_initializer(initializer, stddev)
        activation_fn = tf.nn.relu if activation_fn == 'relu' else None

        with tf.variable_scope(name + '_conv2d') as scope:
            if w is None:
                w = tf.get_variable("weight", shape=[size, size, shape[1], filters], initializer=initializer)
                self.weights.append(w)
            if b is None:
                b = tf.Variable(tf.constant(bias_start, shape=[filters]), name="bias")
                self.biases.append(b)

            c = tf.nn.conv2d(source, w, strides=[1, 1, stride, stride], padding=padding, data_format='NCHW')
            out = tf.nn.bias_add(c, b, data_format='NCHW')
            activated = activation_fn(out) if activation_fn is not None else out
            self.activations.append(activated)

            return activated, w, b

    def float(self, shape, name='float'):
        return tf.placeholder('float32', shape, name=name)

    def int(self, shape, name='int'):
        return tf.placeholder('int64', shape, name=name)

    def q(self, states):
        return self.sess.run([self.q_action, self.output] + self.additional_q_ops, feed_dict={self.state: states})

    def train(self, states, actions, terminals, next_states, rewards, lookahead=None, target_network=None):

        network = self if target_network is None else target_network

        data = {
            self.state: states,
            self.action: actions,
            self.terminal: terminals,
            self.reward: rewards,
            self.next_q: self.sess.run(network.q_max, feed_dict={network.state: next_states})
        }

        if lookahead is not None:
            data[self.lookahead] = lookahead

        _, self.training_iterations, delta, self.batch_loss, self.lr = self.sess.run(
            [self.optimizer, self.global_step, self.delta, self.loss, self.learning_rate], feed_dict=data)

        return delta, self.batch_loss

class Baseline(Network):
    def __init__(self, args, environment, name='baseline_network', sess=None):
        with tf.variable_scope(name) as scope:
            Network.__init__(self, args, environment, name, sess)

            # Build Network
            self.conv1,  w1, b1 = self.conv2d(self.state, size=8, filters=32, stride=4, name='conv1')
            self.conv2,  w2, b2 = self.conv2d(self.conv1, size=4, filters=64, stride=2, name='conv2')
            self.conv3,  w3, b3 = self.conv2d(self.conv2, size=3, filters=64, stride=1, name='conv3')
            self.fc4,    w4, b4 = self.linear(self.flatten(self.conv3, name="fc4"), 512, name='fc4')
            self.output, w5, b5 = self.linear(self.fc4, environment.get_num_actions(), activation_fn='none', name='output')

            self.post_init()


class Linear(Network):
    def __init__(self, args, environment, name='linear_network', sess=None):
        with tf.variable_scope(name) as scope:
            Network.__init__(self, args, environment, name, sess)

            self.fc1,    w1, b1 = self.linear(self.flatten(self.state, name="fc1"), 500, name='fc1')
            self.fc2,    w2, b2 = self.linear(self.fc1, 500, name='fc2')
            self.output, w2, b2 = self.linear(self.fc2, environment.get_num_actions(), activation_fn='none', name='output')

            self.post_init()

class Constrained(Network):
    def __init__(self, args, environment, name='constrained_network', sess=None):
        with tf.variable_scope(name) as scope:
            Network.__init__(self, args, environment, name, sess)

            self.conv1,     w1, b1 = self.conv2d(self.state, size=8, filters=32, stride=4, name='conv1')
            self.conv2,     w2, b2 = self.conv2d(self.conv1, size=4, filters=64, stride=2, name='conv2')
            self.conv3,     w3, b3 = self.conv2d(self.conv2, size=3, filters=64, stride=1, name='conv3')
            self.fc4,       w4, b4 = self.linear(self.flatten(self.conv3), 256, name='fc4')

            self.h,         w5, b5 = self.linear(self.fc4, 256, name='h')
            self.h1,        w6, b6 = self.linear(self.h, 256, name='h1')
            self.hhat,      w7, b7 = self.linear(self.h1, 256, name='hhat')

            self.fc8,       w8, b8 = self.linear(self.merge(self.h, self.hhat, name="fc8"), 256, name='fc8')
            self.output,    w9, b9 = self.linear(self.fc8, environment.get_num_actions(), activation_fn='none', name='output')

            self.hhat_conv1, _, _ = self.conv2d(self.lookahead, size=8, filters=32, stride=4, name='hhat_conv1', w=w1, b=b1)
            self.hhat_conv2, _, _ = self.conv2d(self.hhat_conv1, size=4, filters=64, stride=2, name='hhat_conv2', w=w2, b=b2)
            self.hhat_conv3, _, _ = self.conv2d(self.hhat_conv2, size=3, filters=64, stride=1, name='hhat_conv3', w=w3, b=b3)
            self.hhat_truth, _, _ = self.linear(self.flatten(self.hhat_conv3), 256, name='hhat_fc4', w=w4, b=b4)

            with tf.name_scope("constraint_error") as _:
                self.constraint_error = tf.reduce_mean(tf.pow(tf.sub(self.hhat, self.hhat_truth), 2), reduction_indices=1)

            self.post_init()

    def get_loss(self, processed_delta, prediction, truth):
        return tf.reduce_mean(tf.square(processed_delta)) + tf.reduce_mean(self.constraint_error)


class Density(Network):
    def __init__(self, args, environment, name='mdn_network', sess=None):
        with tf.variable_scope(name) as scope:
            Network.__init__(self, args, environment, name, sess)

            # Build Network
            self.conv1,    w1, b1 = self.conv2d(self.state, size=8, filters=32, stride=4, name='conv1')
            self.conv2,    w2, b2 = self.conv2d(self.conv1, size=4, filters=64, stride=2, name='conv2')
            self.conv3,    w3, b3 = self.conv2d(self.conv2, size=3, filters=64, stride=1, name='conv3')
            self.fc4,      w4, b4 = self.linear(self.flatten(self.conv3, name="fc4"), 512, name='fc4')
            self.output,   w5, b5 = self.linear(self.fc4, environment.get_num_actions(), activation_fn='none', name='output')
            self.variance, w6, b6 = self.linear(self.fc4, environment.get_num_actions(), activation_fn='sigmoid', name='variance')

            self.additional_q_ops.append(self.variance)

            self.post_init()

    def get_loss(self, processed_delta, prediction, truth):
        sigma = self.sum(self.variance * self.action_one_hot, name='variance_acted') + 0.01
        return tf.reduce_mean(tf.log(sigma) + tf.square(processed_delta) / (2.0 * tf.square(sigma)))


class Causal(Network):
    def __init__(self, args, environment, name='baseline_network', sess=None):
        with tf.variable_scope(name) as scope:
            Network.__init__(self, args, environment, name, sess)

            # Common Perception
            self.l1,     w1, b1 = self.conv2d(self.state, size=8, filters=32, stride=4, name='conv1')

            # A Side
            self.l2a,    w2, b2 = self.conv2d(self.l1, size=4, filters=64, stride=2, name='a_conv2')
            self.l2a_fc, w3, b3 = self.linear(self.flatten(self.l2a, name="a_fc4"), 32, activation_fn='none', name='a_fc3')

            # B Side
            self.l2b,    w4, b4 = self.conv2d(self.l1, size=4, filters=64, stride=2, name='b_conv2')
            self.l2b_fc, w5, b5 = self.linear(self.flatten(self.l2b, name="b_fc4"), 32, activation_fn='none', name='b_fc3')

            # Causal Matrix
            self.l2a_fc_e = self.expand(self.l2a_fc, 2, name='a')  # now ?x32x1
            self.l2b_fc_e = self.expand(self.l2b_fc, 1, name='b')  # now ?x1x32
            self.causes = self.flatten(tf.batch_matmul(self.l2a_fc_e, self.l2b_fc_e, name='causes'))

            self.l4,      w6, b6 = self.linear(self.causes, 512, activation_fn='sigmoid', name='l4')
            self.output,  w5, b5 = self.linear(self.l4, environment.get_num_actions(), activation_fn='none', name='output')

            self.post_init()
