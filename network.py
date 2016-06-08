import tensorflow as tf
import subprocess
import os
import numpy as np
import shutil


class Network:
    def __init__(self, args, environment):
        self.args = args
        self.environment = environment
        self.default_initializer = args.initializer

    def one_hot(self, source, size, name='onehot'):
        return tf.one_hot(tf.cast(source, 'int64'), size, 1.0, 0.0, name=name)

    def argmax(self, source, name='argmax'):
        return tf.argmax(source, dimension=1)

    def max(self, source, name='max'):
        return tf.reduce_max(source, reduction_indices=1, name=name)

    def flatten(self, source, name='flatten'):
        shape = source.get_shape().as_list()
        dim = reduce(lambda x, y: x * y, shape[1:])

        with tf.variable_scope(name + "_flatten"):
            return tf.reshape(source, [-1, dim], name=name)

    def expand(self, source, dim, name='expand'):
        with tf.variable_scope(name + "_expand"):
            return tf.expand_dims(source, dim)

    def merge(self, left, right, idx=1, name='merge'):
        with tf.variable_scope(name + "_merge"):
            return tf.concat(idx, [left, right])

    def sum(self, source, name=None, idx=1):
        return tf.reduce_sum(source, reduction_indices=idx, name=name)

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

    def linear(self, source, output_size, stddev=0.02, initializer='default', bias_start=0.01, activation_fn='relu',
               name='linear'):
        shape = source.get_shape().as_list()

        initializer = self.parse_initializer(initializer, stddev)
        activation_fn = self.parse_activation(activation_fn)

        with tf.variable_scope(name + '_linear') as scope:
            w = tf.get_variable("weight", [shape[1], output_size], tf.float32, initializer)
            b = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))

            out = tf.nn.bias_add(tf.matmul(source, w), b)
            activated = activation_fn(out) if activation_fn is not None else out

            return activated, w, b

    def conv2d(self, source, size, filters, stride, padding='SAME', stddev=0.02, initializer='default', bias_start=0.01,
               activation_fn='relu', name='conv2d'):
        shape = source.get_shape().as_list()
        initializer = self.parse_initializer(initializer, stddev)
        activation_fn = self.parse_activation(activation_fn)

        with tf.variable_scope(name + '_conv2d') as scope:
            w = tf.get_variable("weight", shape=[size, size, shape[1], filters], initializer=initializer)
            b = tf.get_variable("bias", [filters], initializer=tf.constant_initializer(bias_start))

            c = tf.nn.conv2d(source, w, strides=[1, 1, stride, stride], padding=padding, data_format='NCHW')
            out = tf.nn.bias_add(c, b, data_format='NCHW')
            activated = activation_fn(out) if activation_fn is not None else out

            return activated, w, b

    def float(self, shape, name='float'):
        return tf.placeholder('float16', shape, name=name)

    def to_float(self, source):
        return tf.cast(source, 'float16')

    def int(self, shape, name='int', bits=8):
        return tf.placeholder('int' + str(bits), shape, name=name)

    def loss(self, processed_delta, prediction, truth):
        return tf.reduce_mean(tf.square(processed_delta, name='square'), name='loss')

    def optimizer(self, learning_rate):
        return tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                         decay=self.args.rms_decay,
                                         momentum=float(self.args.rms_momentum),
                                         epsilon=self.args.rms_eps)

    def environment_scale(self, states):
        return tf.truediv(tf.to_float(states), tf.to_float(self.environment.max_state_value()))

class Commander(Network):
    def __init__(self, Type, args, environment):
        Network.__init__(self, args, environment)

        self.training_iterations = 0
        self.batch_loss = 0
        self.lr = 0
        gradients = []

        self.sess = self.start_session(args)
        network = Type(args, environment)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        with tf.name_scope('learning_rate'), tf.device('/gpu:1'):
            self.learning_rate = tf.maximum(self.args.learning_rate_end,
                                               tf.train.exponential_decay(
                                                 self.args.learning_rate_start,
                                                 self.global_step,
                                                 self.args.learning_rate_decay_step,
                                                 self.args.learning_rate_decay,
                                                 staircase=False))

        with tf.device('/gpu:1'):
            optimizer = network.optimizer(learning_rate=self.learning_rate)

        with tf.name_scope('inputs') as _:
            self.states = self.int([None, args.phi_frames] + list(environment.get_state_space()), name='state')
            self.next_states = self.int([None, args.phi_frames] + list(environment.get_state_space()), name='next_state')
            self.actions = self.int([None], name='action_index')
            self.terminals = self.int([None], name='terminal')
            self.rewards = self.int([None], name='reward', bits=32)

        with tf.device('/gpu:0'):
            with tf.variable_scope('target_network'):  # Target network variables were created by thread_actor
                next_qs = network.build(states=self.environment_scale(self.next_states))
                next_best_qs = self.max(next_qs)

            with tf.variable_scope('train_network'):
                train_qs = network.build(states=self.environment_scale(self.states))
                train_acted_qs = self.sum(train_qs * self.one_hot(self.actions, environment.get_num_actions()))

            with tf.device('/gpu:1'):
                with tf.name_scope('target_q'):
                    processed_rewards = tf.clip_by_value(self.rewards, -1, 1, name='clip_reward') if self.args.clip_reward else self.rewards
                    target_q = tf.stop_gradient(self.to_float(processed_rewards) + self.to_float(self.args.discount) * (self.to_float(1.0) - self.to_float(self.terminals)) * self.to_float(next_best_qs))

                with tf.name_scope('delta'):
                    delta = self.to_float(target_q) - self.to_float(train_acted_qs)
                    processed_delta = tf.clip_by_value(delta, -1.0, 1.0) if self.args.clip_tderror else delta

                with tf.name_scope('loss'):
                    self.loss_op = network.loss(processed_delta, prediction=train_acted_qs, truth=target_q)
                    self.tderror = tf.pow(delta, self.args.priority_temperature)

                gradient = optimizer.compute_gradients(self.loss_op)
                # tf.truediv(grad, tf.to_float(self.args.towers))
                gradients += [(grad, var) for grad, var in gradient if grad is not None]

            with tf.name_scope('thread_actor'), tf.variable_scope('target_network', reuse=True):
                self.qs = network.build(states=self.environment_scale(self.states))
                self.qs_argmax = self.argmax(self.qs)

        self.target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_network')
        self.train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='train_network')

        with tf.name_scope('copy'):
            self.assign_ops = [target.assign(train) for target, train in zip(self.target_vars, self.train_vars)]

        with tf.device('/gpu:1'):
            self.train_op = optimizer.apply_gradients(gradients, global_step=self.global_step)

        init_op = tf.initialize_all_variables()

        self.sess.run(init_op)
        self.tensorboard()

    def __len__(self):
        return sum([sum([reduce(lambda x, y: x * y, l.get_shape().as_list()) for l in e]) for e in [self.train_vars]])

    def start_session(self, args):
        tf.set_random_seed(args.random_seed)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction,
                                    allow_growth=True)

        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                                                allow_soft_placement=True,
                                                log_device_placement=self.args.verbose))


    def tensorboard(self):
        tf.train.SummaryWriter(self.args.tf_summary_path, self.sess.graph)
        self.tensorboard_process = subprocess.Popen(["tensorboard", "--logdir=" + self.args.tf_summary_path],
                         stdout=open(os.devnull, 'w'),
                         stderr=open(os.devnull, 'w'),
                         close_fds=True)

    def update(self):
        self.sess.run(self.assign_ops)

    def train(self, states, actions, terminals, next_states, rewards, lookahead=None):

        data = {
            self.states: states,
            self.actions: actions,
            self.terminals: terminals,
            self.rewards: rewards,
            self.next_states: next_states
        }

        _, error, self.batch_loss, self.lr, self.training_iterations = self.sess.run([self.train_op,
                                                                                      self.tderror,
                                                                                      self.loss_op,
                                                                                      self.learning_rate,
                                                                                      self.global_step],
                                                                                      feed_dict=data)

        if self.training_iterations % self.args.copy_frequency == 0:
            self.update()

        return error, self.batch_loss

    def q(self, states):
        return self.sess.run([self.qs_argmax, self.qs], feed_dict={self.states: states})

class Convergence(Commander):
    def __init__(self, Type, args, environment):
        assert args.agent_type == 'convergence', 'Convergence Commander must use Convergence Agent'
        Commander.__init__(self, Type, args, environment)

        # self.clear_ops = []
        # vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='train_network')

        # for var in vars:
            # newvar = (1 - mask) * variables + mask * random
            # self.clear_ops.append(var.assign(newvar))

    def clear(self):
        self.sess.run(self.clear_ops)

    def train(self, states, actions, terminals, next_states, rewards, lookahead=None):

        data = {
            self.states: states,
            self.actions: actions,
            self.terminals: terminals,
            self.rewards: rewards,
            self.next_states: next_states
        }

        _, error, self.batch_loss, self.lr, self.training_iterations = self.sess.run([self.train_op,
                                                                               self.tderror,
                                                                               self.loss_op,
                                                                               self.learning_rate,
                                                                               self.global_step],
                                                                              feed_dict=data)

        return error, self.batch_loss

class Baseline(Network):
    def __init__(self, args, environment):
        Network.__init__(self, args, environment)

    def build(self, states):
        conv1, w1, b1 = self.conv2d(states, size=8, filters=32, stride=4, name='conv1')
        conv2, w2, b2 = self.conv2d(conv1, size=4, filters=64, stride=2, name='conv2')
        conv3, w3, b3 = self.conv2d(conv2, size=3, filters=64, stride=1, name='conv3')
        fc4, w4, b4 = self.linear(self.flatten(conv3, name="fc4"), 512, name='fc4')
        output, w5, b5 = self.linear(fc4, self.environment.get_num_actions(), activation_fn='none', name='output')

        return output


class Linear(Network):
    def __init__(self, args, environment):
        Network.__init__(self, args, environment)

    def build(self, states):
        fc1,    w1, b1 = self.linear(self.flatten(states, name="fc1"), 500, name='fc1')
        fc2,    w2, b2 = self.linear(fc1, 500, name='fc2')
        output, w2, b2 = self.linear(fc2, self.environment.get_num_actions(), activation_fn='none', name='output')

        return output


class Constrained(Network):
    def __init__(self, args, environment):
        Network.__init__(self, args, environment)

    def build(self, states):
        conv1,     w1, b1 = self.conv2d(states, size=8, filters=32, stride=4, name='conv1')
        conv2,     w2, b2 = self.conv2d(conv1, size=4, filters=64, stride=2, name='conv2')
        conv3,     w3, b3 = self.conv2d(conv2, size=3, filters=64, stride=1, name='conv3')
        fc4,       w4, b4 = self.linear(self.flatten(conv3), 256, name='fc4')

        h,         w5, b5 = self.linear(fc4, 256, name='h')
        h1,        w6, b6 = self.linear(h, 256, name='h1')
        hhat,      w7, b7 = self.linear(h1, 256, name='hhat')

        fc8,       w8, b8 = self.linear(self.merge(h, hhat, name="fc8"), 256, name='fc8')
        output,    w9, b9 = self.linear(fc8, self.environment.get_num_actions(), activation_fn='none', name='output')

        # hhat_conv1, _, _ = self.conv2d(lookahead, size=8, filters=32, stride=4, name='hhat_conv1', w=w1, b=b1)
        # hhat_conv2, _, _ = self.conv2d(s.hhat_conv1, size=4, filters=64, stride=2, name='hhat_conv2', w=w2, b=b2)
        # hhat_conv3, _, _ = self.conv2d(s.hhat_conv2, size=3, filters=64, stride=1, name='hhat_conv3', w=w3, b=b3)
        # hhat_truth, _, _ = self.linear(s.flatten(self.hhat_conv3), 256, name='hhat_fc4', w=w4, b=b4)

        # with tf.name_scope("constraint_error") as _:
        #     self.constraint_error = tf.reduce_mean(tf.pow(tf.sub(self.hhat, self.hhat_truth), 2), reduction_indices=1)

        return output

    def loss(self, processed_delta, prediction, truth):
        return tf.reduce_mean(tf.square(processed_delta)) #+ tf.reduce_mean(self.constraint_error)


class Density(Network):
    def __init__(self, args, environment):
        Network.__init__(self, args, environment)

    def build(self, states):

        # Build Network
        conv1,    w1, b1 = self.conv2d(states, size=8, filters=32, stride=4, name='conv1')
        conv2,    w2, b2 = self.conv2d(conv1, size=4, filters=64, stride=2, name='conv2')
        conv3,    w3, b3 = self.conv2d(conv2, size=3, filters=64, stride=1, name='conv3')
        fc4,      w4, b4 = self.linear(self.flatten(conv3, name="fc4"), 512, name='fc4')
        output,   w5, b5 = self.linear(fc4, self.environment.get_num_actions(), activation_fn='none', name='output')
        variance, w6, b6 = self.linear(fc4, self.environment.get_num_actions(), activation_fn='sigmoid', name='variance')

        #self.additional_q_ops.append(self.variance)

        return output

    # def loss(self, processed_delta, prediction, truth):
    #     sigma = self.sum(self.variance * self.action_one_hot, name='variance_acted') + 0.0001
    #     return tf.reduce_mean(tf.log(sigma) + tf.square(processed_delta) / (2.0 * tf.square(sigma)))


class Causal(Network):
    def __init__(self, args, environment):
        Network.__init__(self, args, environment)

    def build(self, states):
        # Common Perception
        l1,     w1, b1 = self.conv2d(states, size=8, filters=32, stride=4, name='conv1')

        # A Side
        l2a,    w2, b2 = self.conv2d(l1, size=4, filters=64, stride=2, name='a_conv2')
        l2a_fc, w3, b3 = self.linear(self.flatten(l2a, name="a_fc4"), 32, activation_fn='none', name='a_fc3')

        # B Side
        l2b,    w4, b4 = self.conv2d(l1, size=4, filters=64, stride=2, name='b_conv2')
        l2b_fc, w5, b5 = self.linear(self.flatten(l2b, name="b_fc4"), 32, activation_fn='none', name='b_fc3')

        # Causal Matrix
        l2a_fc_e = self.expand(l2a_fc, 2, name='a')  # now ?x32x1
        l2b_fc_e = self.expand(l2b_fc, 1, name='b')  # now ?x1x32
        causes = self.flatten(tf.batch_matmul(l2a_fc_e, l2b_fc_e, name='causes'))

        l4,      w6, b6 = self.linear(causes, 512, name='l4')
        output,  w5, b5 = self.linear(l4, self.environment.get_num_actions(), activation_fn='none', name='output')

        return output
