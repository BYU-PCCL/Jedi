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
        return tf.one_hot(source, size, 1.0, 0.0, name=name)

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
               name='linear', w=None, b=None):
        shape = source.get_shape().as_list()

        initializer = self.parse_initializer(initializer, stddev)
        activation_fn = self.parse_activation(activation_fn)

        with tf.variable_scope(name + '_linear') as scope:
            if w is None:
                w = tf.get_variable("weight", [shape[1], output_size], tf.float32, initializer)
            if b is None:
                b = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))

            out = tf.nn.bias_add(tf.matmul(source, w), b)
            activated = activation_fn(out) if activation_fn is not None else out

            return activated, w, b

    def conv2d(self, source, size, filters, stride, padding='SAME', stddev=0.02, initializer='default', bias_start=0.01,
               activation_fn='relu', name='conv2d', w=None, b=None):
        shape = source.get_shape().as_list()
        initializer = self.parse_initializer(initializer, stddev)
        activation_fn = tf.nn.relu if activation_fn == 'relu' else None

        with tf.variable_scope(name + '_conv2d') as scope:
            if w is None:
                w = tf.get_variable("weight", shape=[size, size, shape[1], filters], initializer=initializer)
            if b is None:
                b = tf.get_variable("bias", [filters], initializer=tf.constant_initializer(bias_start))

            c = tf.nn.conv2d(source, w, strides=[1, 1, stride, stride], padding=padding, data_format='NCHW')
            out = tf.nn.bias_add(c, b, data_format='NCHW')
            activated = activation_fn(out) if activation_fn is not None else out

            return activated, w, b

    def float(self, shape, name='float'):
        return tf.placeholder('float32', shape, name=name)

    def int(self, shape, name='int'):
        return tf.placeholder('int64', shape, name=name)

    def loss(self, delta):
        return tf.reduce_mean(tf.square(delta), name='loss')

    def optimizer(self, learning_rate):
        return tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                         decay=self.args.rms_decay,
                                         momentum=float(self.args.rms_momentum),
                                         epsilon=self.args.rms_eps)

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
        optimizer = network.optimizer(learning_rate=self.args.learning_rate_end)

        with tf.name_scope('inputs') as _:
            self.states = self.float([None, args.phi_frames] + list(environment.get_state_space()), name='state')
            self.next_states = self.float([None, args.phi_frames] + list(environment.get_state_space()), name='state')
            self.actions = self.int([None], name='action_index')
            self.terminals = self.float([None], name='terminal')
            self.rewards = self.float([None], name='reward')

        with tf.name_scope('split_inputs') as _:
            assert self.args.batch_size % self.args.threads == 0, "Error: Threads must divide batch_size evenly"
            states = tf.split(split_dim=0, num_split=self.args.threads, value=self.states)
            actions = tf.split(split_dim=0, num_split=self.args.threads, value=self.actions)
            terminals = tf.split(split_dim=0, num_split=self.args.threads, value=self.terminals)
            next_states = tf.split(split_dim=0, num_split=self.args.threads, value=self.next_states)
            rewards = tf.split(split_dim=0, num_split=self.args.threads, value=self.rewards)

        for n in range(self.args.threads):
            with tf.device('/gpu:{}'.format(n % 2)):
                with tf.name_scope('thread_{}'.format(n)):

                    processed_rewards = tf.clip_by_value(rewards[n], -1.0, 1.0, name='clip_reward') if self.args.clip_reward else rewards[n]

                    with tf.variable_scope('target_network', reuse=n > 0) as scope:
                        next_qs = network.build(states=next_states[n] / float(environment.max_state_value()))
                        next_best_qs = self.max(next_qs)

                    with tf.variable_scope('train_network', reuse=n > 0) as scope:
                        train_qs = network.build(states=states[n] / float(environment.max_state_value()))
                        train_acted_qs = self.sum(train_qs * self.one_hot(actions[n], environment.get_num_actions()))

                    with tf.name_scope('target_q'):
                        target_q = tf.stop_gradient(processed_rewards + self.args.discount * (1 - terminals[n]) * next_best_qs)

                    with tf.name_scope('delta'):
                        delta = train_acted_qs - target_q
                        processed_delta = tf.clip_by_value(delta, -1.0, 1.0) if self.args.clip_tderror else delta

                    with tf.name_scope('loss'):
                        loss = network.loss(processed_delta)

                    gradient = optimizer.compute_gradients(loss, colocate_gradients_with_ops=True)
                    gradients.append(gradient)

        with tf.name_scope('thread_actor'):
            with tf.variable_scope('target_network', reuse=True) as _:
                self.qs = network.build(states=self.states)
                self.qs_argmax = self.argmax(self.qs)

        self.target_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='target')
        self.train_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='train')

        with tf.name_scope('copy'):
            self.assign_ops = [target.assign(train) for target, train in zip(self.target_vars, self.train_vars)]

        gradient = self.average_gradients(gradients)
        self.train_op = optimizer.apply_gradients(gradient, global_step=self.global_step)

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
                                                allow_soft_placement=True))

    def tensorboard(self):
        tf.train.SummaryWriter(self.args.tf_summary_path, self.sess.graph)
        subprocess.Popen(["tensorboard", "--logdir=" + self.args.tf_summary_path],
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

        _, self.training_iterations = self.sess.run([self.train_op, self.global_step], feed_dict=data)

        if self.training_iterations % self.args.copy_frequency == 0:
            self.update()

    def q(self, states):
        return self.sess.run([self.qs_argmax, self.qs], feed_dict={self.states: states})

    def average_gradients(self, tower_grads):
        with tf.name_scope('average_gradients'):

            average_grads = []
            for grad_and_vars in zip(*tower_grads):

                # Note that each grad_and_vars looks like the following:
                #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
                grads = []
                for gradient, variable in grad_and_vars:
                    if gradient is not None:
                        expanded_g = tf.expand_dims(gradient, 0) # add tower dim
                        grads.append(expanded_g)

                if len(grads) > 0:
                    # Average over the 'tower' dimension.
                    grad = tf.concat(0, grads)
                    grad = tf.reduce_mean(grad, 0)

                    # Keep in mind that the Variables are redundant because they are shared
                    # across towers. So .. we will just return the first tower's pointer to
                    # the Variable.
                    v = grad_and_vars[0][1]
                    grad_and_var = (grad, v)
                    average_grads.append(grad_and_var)
            return average_grads



class Baseline(Network):
    def __init__(self, args, environment):
        Network.__init__(self, args, environment)

    def build(self, states):
        # Build Network
        conv1, w1, b1 = self.conv2d(states, size=8, filters=32, stride=4, name='conv1')
        conv2, w2, b2 = self.conv2d(conv1, size=4, filters=64, stride=2, name='conv2')
        conv3, w3, b3 = self.conv2d(conv2, size=3, filters=64, stride=1, name='conv3')
        fc4, w4, b4 = self.linear(self.flatten(conv3, name="fc4"), 512, name='fc4')
        output, w5, b5 = self.linear(fc4, self.environment.get_num_actions(), activation_fn='none', name='output')

        return output


class Linear(Network):
    def __init__(self, args, environment, name='network', sess=None):
        with tf.variable_scope("linear-" + name) as scope:
            Network.__init__(self, args, environment, name, sess)

            self.fc1,    w1, b1 = self.linear(self.flatten(self.state, name="fc1"), 500, name='fc1')
            self.fc2,    w2, b2 = self.linear(self.fc1, 500, name='fc2')
            self.output, w2, b2 = self.linear(self.fc2, environment.get_num_actions(), activation_fn='none', name='output')

            self.post_init()


class Constrained(Network):
    def __init__(self, args, environment, name='network', sess=None):
        with tf.variable_scope("constrained-" + name) as scope:
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
    def __init__(self, args, environment, name='network', sess=None):
        with tf.variable_scope("density-" + name) as scope:
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
        sigma = self.sum(self.variance * self.action_one_hot, name='variance_acted') + 0.0001
        return tf.reduce_mean(tf.log(sigma) + tf.square(processed_delta) / (2.0 * tf.square(sigma)))


class Causal(Network):
    def __init__(self, args, environment, name='network', sess=None):
        with tf.variable_scope("causal-" + name) as scope:
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

            self.l4,      w6, b6 = self.linear(self.causes, 512, name='l4')
            self.output,  w5, b5 = self.linear(self.l4, environment.get_num_actions(), activation_fn='none', name='output')

            self.post_init()
