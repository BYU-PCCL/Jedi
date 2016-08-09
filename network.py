import tensorflow as tf
import subprocess
import os
import numpy as np
import ops as op
import random
from functools import reduce

class DQN(object):
    def __init__(self, Type, args, environment):
        self.args = args
        self.environment = environment
        self.tensorboard_process = None

        self.training_iterations = 0
        self.batch_loss = 0
        self.learning_rate = 0
        gradients = []

        self.sess = self.start_session(args)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        with op.context(floatx=tf.float16, floatsafe=False):

            inputs = Type.Inputs(args, environment)

            with tf.device('/gpu:0'):
                with tf.variable_scope('target_network'):
                    target_network = Type(args, environment, inputs)
                    target_output_next_states = target_network.build(inputs.next_states)

                with tf.variable_scope('train_network'):
                    self.train_network = Type(args, environment, inputs)
                    self.train_output_states = self.train_network.build(inputs.states)

                with tf.variable_scope('train_network', reuse=True):
                    train_output_next_states = self.train_network.build(inputs.next_states)

            with tf.device('/gpu:1'):
                with tf.name_scope('thread_actor'), tf.variable_scope('target_network', reuse=True):
                    self.actor_network = Type(args, environment, inputs)
                    self.actor_output = self.actor_network.build(inputs.states)
                    self.actor_output_action = self.actor_network.action(self.actor_output)
                    self.testop = inputs.rewards

            with tf.device('/gpu:1'):
                with tf.name_scope('loss'):
                    truth = tf.stop_gradient(self.train_network.truth(train_output_states=self.train_output_states,
                                                                      train_output_next_states=train_output_next_states,
                                                                      target_output_next_states=target_output_next_states))
                    prediction = self.train_network.prediction(train_output_states=self.train_output_states)

                    self.loss_op = self.train_network.loss(truth=truth, prediction=prediction)
                    self.priority_op = self.train_network.priority(truth=truth, prediction=prediction)

                # It's about 2x faster for us to compute/apply the gradients than to use optimizer.minimize()
                with tf.name_scope('optimizer'):
                    self.learning_rate_op = self.train_network.learning_rate(step=self.global_step)
                    optimizer = self.train_network.optimizer(self.learning_rate_op)
                    gradient = optimizer.compute_gradients(self.loss_op)
                    gradients += [(grad, var) for grad, var in gradient if grad is not None]
                    self.train_op = optimizer.apply_gradients(gradients, global_step=self.global_step)

                self.target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_network')
                self.train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='train_network')
                self.assign_ops = [target.assign(train) for target, train in zip(self.target_vars, self.train_vars)]

                self.inputs = inputs
        self.initialize_op = tf.initialize_all_variables()
        self.initialize()
        self.tensorboard()

    def initialize(self):
        self.sess.run(self.initialize_op)

    def total_parameters(self):
        return sum([sum([reduce(lambda x, y: x * y, l.get_shape().as_list()) for l in e]) for e in [self.train_vars]])

    def start_session(self, args):
        tf.set_random_seed(args.random_seed)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction,
                                    allow_growth=True)

        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                                                allow_soft_placement=True,
                                                log_device_placement=self.args.verbose))


    def tensorboard(self):
        if self.tensorboard_process is not None:
            self.tensorboard_process.kill()
        tf.train.SummaryWriter(self.args.tf_summary_path, self.sess.graph)
        self.tensorboard_process = subprocess.Popen(["tensorboard", "--logdir=" + self.args.tf_summary_path],
                                                    stdout=open(os.devnull, 'w'),
                                                    stderr=open(os.devnull, 'w'),
                                                    close_fds=True)

    def update(self):
        if (self.training_iterations + 1) % self.args.copy_frequency == 0:
            self.sess.run(self.assign_ops)

    def build_feed_dict(self, **kwargs):
        # Print a notice to the user if they are passing in unknown variables
        ignored = [key for key in kwargs.keys() if key + '_placeholder' not in self.inputs.__dict__.keys()]
        assert len(ignored) == 0, 'The following arguments passed to train() are not used by this network : ' + str(ignored)
        return {getattr(self.inputs, key + '_placeholder'): var for (key, var) in kwargs.items()}

    def train(self, **kwargs):
        data = self.build_feed_dict(**kwargs)

        self.train_network.debug(self.sess, data)
        _, priority, self.batch_loss, self.learning_rate, self.training_iterations = self.sess.run([self.train_op,
                                                                                                    self.priority_op,
                                                                                                    self.loss_op,
                                                                                                    self.learning_rate_op,
                                                                                                    self.global_step],
                                                                                                   feed_dict=data)

        self.update()

        return priority, self.batch_loss

    def q(self, **kwargs):
        data = self.build_feed_dict(**kwargs)
        results = self.sess.run([self.actor_output_action, self.actor_output] + self.actor_network.additional_q_ops,
                                feed_dict=data)

        return results[0], results[1], results[2:]


class Network(object):

    class Inputs:
        def __getattr__(self, item):
            processed = {'actions': self.actions_placeholder,
                         'rewards': op.optional_clip(self.rewards_placeholder, -1, 1, self.args.clip_reward),
                         'states': op.environment_scale(self.states_placeholder, self.environment),
                         'next_states': op.environment_scale(self.next_states_placeholder, self.environment),
                         'terminals': self.terminals_placeholder}

            processed.update(self.preprocessing())

            return processed[item]

        def __init__(self, args, environment):
            self.args = args
            self.environment = environment

            with tf.name_scope('inputs'):
                self.states_placeholder = op.int([None, args.phi_frames] + list(environment.get_state_space()), name='state', unsigned=True)
                self.next_states_placeholder = op.int([None, args.phi_frames] + list(environment.get_state_space()), name='next_state', unsigned=True)
                self.actions_placeholder = op.int([None], name='action_index', unsigned=True)
                self.terminals_placeholder = op.int([None], name='terminal')
                self.rewards_placeholder = op.int([None], name='reward', bits=32)

        def preprocessing(self):
            return {}

    def __init__(self, args, environment, inputs):
        self.args = args
        self.environment = environment
        self.inputs = inputs
        self.additional_q_ops = []

    def build(self, states):
        pass

    def truth(self, train_output_states, train_output_next_states, target_output_next_states):
        return op.tofloat(self.inputs.rewards) + self.args.discount * (
        1.0 - op.tofloat(self.inputs.terminals)) * op.tofloat(op.max(target_output_next_states))

    def prediction(self, train_output_states):
        return op.get(op.tofloat(train_output_states), self.inputs.actions)

    def loss(self, truth, prediction):
        delta = op.optional_clip(truth - prediction, -1.0, 1.0, self.args.clip_tderror)
        return tf.reduce_mean(tf.square(delta, name='square'), name='loss')

    def priority(self, truth, prediction):
        return tf.pow(truth - prediction, self.args.priority_temperature)

    def optimizer(self, learning_rate):
        return tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                         decay=self.args.rms_decay,
                                         momentum=float(self.args.rms_momentum),
                                         epsilon=self.args.rms_eps)

    def learning_rate(self, step):
        decayed_lr = tf.train.exponential_decay(self.args.learning_rate_start,
                                                step,
                                                self.args.learning_rate_decay_step,
                                                self.args.learning_rate_decay,
                                                staircase=False)
        return tf.maximum(self.args.learning_rate_end, decayed_lr)

    def action(self, output):
        return op.argmax(output)

    def debug(self, session, data):
        pass


class Linear(Network):
    def __init__(self, args, environment, inputs):
        Network.__init__(self, args, environment, inputs)

    def build(self, states):
        with op.    context(default_activation_fn='relu'):
            fc1,    w1, b1 = op.linear(op.flatten(states, name="fc1_flatten"), 500, name='fc1')
            fc2,    w2, b2 = op.linear(fc1, 500, name='fc2')
            value,  w3, b3 = op.linear(fc2, self.environment.get_num_actions(), activation_fn='none', name='value')
            advantages, w4, b4 = op.linear(fc2, self.environment.get_num_actions(), activation_fn='none', name='advantages')

            # Dueling DQN - http://arxiv.org/pdf/1511.06581v3.pdf
            output = value + (advantages - op.mean(advantages, keep_dims=True))

        return output


class Baseline(Network):
    def __init__(self, args, environment, inputs):
        Network.__init__(self, args, environment, inputs)

    def build(self, states):
        with op.context(default_activation_fn='relu'):
            conv1, w1, b1 = op.conv2d(states, size=8, filters=32, stride=4, name='conv1')
            conv2, w2, b2 = op.conv2d(conv1, size=4, filters=64, stride=2, name='conv2')
            conv3, w3, b3 = op.conv2d(conv2, size=3, filters=64, stride=1, name='conv3')
            fc4, w4, b4 = op.linear(op.flatten(conv3, name="fc4"), 512, name='fc4')
            output, w5, b5 = op.linear(fc4, self.environment.get_num_actions(), activation_fn='none', name='output')

            return output


class BaselineDuel(Network):
    def __init__(self, args, environment, inputs):
        Network.__init__(self, args, environment, inputs)

    def build(self, states):
        with op.context(default_activation_fn='relu'):
            conv1, w1, b1 = op.conv2d(states, size=8, filters=32, stride=4, name='conv1')
            conv2, w2, b2 = op.conv2d(conv1, size=4, filters=64, stride=2, name='conv2')
            conv3, w3, b3 = op.conv2d(conv2, size=3, filters=64, stride=1, name='conv3')
            conv3_flatten = op.flatten(conv3, name="conv3_flatten")

            fc4_value, w4, b4 = op.linear(conv3_flatten, 512, name='fc4_value')
            value, w5, b5 = op.linear(fc4_value, 1, activation_fn='none', name='value')

            fc4_advantage, w6, b6 = op.linear(conv3_flatten, 512, name='fc4_advantages')
            advantages, w7, b7 = op.linear(fc4_advantage, self.environment.get_num_actions(), activation_fn='none', name='advantages')

            # Dueling DQN - http://arxiv.org/pdf/1511.06581v3.pdf
            output = value + (advantages - op.mean(advantages, keep_dims=True))

            return output


class BaselineDouble(Baseline):
    def truth(self, train_output_states, train_output_next_states, target_output_next_states):
        # Double DQN - http://arxiv.org/pdf/1509.06461v3.pdf
        double_q_next = tf.stop_gradient(op.get(target_output_next_states, op.argmax(train_output_next_states)))
        return (op.tofloat(self.inputs.rewards) + self.args.discount *
                (1.0 - op.tofloat(self.inputs.terminals)) * op.tofloat(double_q_next))


class BaselineDoubleDuel(BaselineDuel):
    def truth(self, train_output_states, train_output_next_states, target_output_next_states):
        # Double DQN - http://arxiv.org/pdf/1509.06461v3.pdf
        double_q_next = op.get(target_output_next_states, op.argmax(train_output_next_states))
        return (op.tofloat(self.inputs.rewards) + self.args.discount *
                (1.0 - op.tofloat(self.inputs.terminals)) * op.tofloat(double_q_next))


class Constrained(Network):

    class Inputs(Network.Inputs):
        def __init__(self, args, environment):
            Network.Inputs.__init__(self, args, environment)

            with tf.name_scope('network_specific_inputs'):
                self.lookaheads_placeholder = op.int([None, args.phi_frames] + list(environment.get_state_space()),
                                                    name='lookaheads')

        def preprocessing(self):
            return {'lookaheads': op.environment_scale(self.lookaheads_placeholder, self.environment)}

    def __init__(self, args, environment, inputs):
        Network.__init__(self, args, environment, inputs)

    def build(self, states):

        with tf.variable_scope('net'), op.context(default_activation_fn='relu'):
            conv1,     w1, b1 = op.conv2d(states, size=8, filters=32, stride=4, name='conv1')
            conv2,     w2, b2 = op.conv2d(conv1, size=4, filters=64, stride=2, name='conv2')
            conv3,     w3, b3 = op.conv2d(conv2, size=3, filters=64, stride=1, name='conv3')
            fc4,       w4, b4 = op.linear(op.flatten(conv3), 256, name='fc4')

            h,         w5, b5 = op.linear(fc4, 256, name='h')
            h1,        w6, b6 = op.linear(h, 256, name='h1')
            hhat,      w7, b7 = op.linear(h1, 256, name='hhat')

            fc8,       w8, b8 = op.linear(op.merge(h, hhat, name="fc8"), 256, name='fc8')
            output,    w9, b9 = op.linear(fc8, self.environment.get_num_actions(), activation_fn='none', name='output')

        with tf.name_scope('prediction'), tf.variable_scope('net', reuse=True), op.context(default_activation_fn='relu'):
            hhat_conv1, _, _ = op.conv2d(self.inputs.lookaheads, size=8, filters=32, stride=4, name='conv1')
            hhat_conv2, _, _ = op.conv2d(hhat_conv1, size=4, filters=64, stride=2, name='conv2')
            hhat_conv3, _, _ = op.conv2d(hhat_conv2, size=3, filters=64, stride=1, name='conv3')
            hhat_truth, _, _ = op.linear(op.flatten(hhat_conv3), 256, name='fc4')

            self.constraint_error = tf.reduce_mean((hhat - hhat_truth)**2, reduction_indices=1, name='prediction_error')

        return output

    def loss(self, truth, prediction):
        delta = op.optional_clip(truth - prediction, -1.0, 1.0, self.args.clip_tderror)
        return tf.reduce_mean(tf.square(delta)) + tf.reduce_mean(op.tofloat(self.constraint_error))


class Density(Network):
    def __init__(self, args, environment, inputs):
        Network.__init__(self, args, environment, inputs)

    def build(self, states):
        with op.context(default_activation_fn='relu'):
            conv1,    w1, b1 = op.conv2d(states, size=8, filters=32, stride=4, name='conv1')
            conv2,    w2, b2 = op.conv2d(conv1, size=4, filters=64, stride=2, name='conv2')
            conv3,    w3, b3 = op.conv2d(conv2, size=3, filters=64, stride=1, name='conv3')
            fc4,      w4, b4 = op.linear(op.flatten(conv3, name="fc4"), 512, name='fc4')
            output,   w5, b5 = op.linear(fc4, self.environment.get_num_actions(), activation_fn='none', name='output')
            raw_sigma, w6, b6 = op.linear(fc4, self.environment.get_num_actions(), name='variance')

            raw_sigma += 0.0001  # to avoid divide by zero
            self.sigma = tf.exp(raw_sigma)
            self.additional_q_ops.append(self.sigma)

        return output

    def loss(self, truth, prediction):
        y = prediction
        mu = truth
        sigma = op.get(self.sigma, self.inputs.actions)

        # Gaussian log-likelihood
        result = op.tofloat(y - mu)  # Primarily to prevent under/overflow since they are already float16
        result = tf.cast(result, 'float32') * tf.inv(sigma)
        result = -tf.square(result) / 2
        result = result + tf.log(tf.inv(sigma))

        return tf.reduce_mean(-result)


class Causal(Network):
    def __init__(self, args, environment, inputs):
        Network.__init__(self, args, environment, inputs)

    def build(self, states):
        with op.context(default_activation_fn='relu'):
            # Common Perception
            l1,     w1, b1 = op.conv2d(states, size=8, filters=32, stride=4, name='conv1')

            # A Side
            l2a,    w2, b2 = op.conv2d(l1, size=4, filters=64, stride=2, name='a_conv2')
            l2a_fc, w3, b3 = op.linear(op.flatten(l2a, name="a_fc4"), 32, activation_fn='none', name='a_fc3')

            # B Side
            l2b,    w4, b4 = op.conv2d(l1, size=4, filters=64, stride=2, name='b_conv2')
            l2b_fc, w5, b5 = op.linear(op.flatten(l2b, name="b_fc4"), 32, activation_fn='none', name='b_fc3')

            # Causal Matrix
            l2a_fc_e = op.expand(l2a_fc, 2, name='a')  # now ?x32x1
            l2b_fc_e = op.expand(l2b_fc, 1, name='b')  # now ?x1x32
            causes = op.flatten(tf.batch_matmul(l2a_fc_e, l2b_fc_e, name='causes'))

            l4,      w6, b6 = op.linear(causes, 512, name='l4')
            output,  w5, b5 = op.linear(l4, self.environment.get_num_actions(), activation_fn='none', name='output')

            return output


class ConvergenceDQN(DQN):
    def __init__(self, Type, args, environment):
        DQN.__init__(self, Type, args, environment)

        self.reset_ops = []

        with tf.name_scope('random_reset'):
            for var in self.train_vars:
                with tf.device(var.device):
                    size = np.prod(var.get_shape().as_list())
                    num_reset = int(round(size * args.convergence_percent_reset))

                    indexes = tf.constant(range(size), dtype=tf.int64)  # if dim > 4000000 else tf.uint32
                    indexes = tf.random_shuffle(indexes)
                    indicies_to_reset = tf.slice(indexes, begin=[0], size=[num_reset])

                    self.testvar = var

                    random_values = tf.truncated_normal([num_reset], stddev=.02, dtype=var.dtype.base_dtype) # todo: use var.initializer
                    self.reset_ops.append(tf.scatter_update(var, indices=[indicies_to_reset], updates=[random_values]))

        self.tensorboard()

    def update(self):
        if (self.training_iterations + 1) % self.args.copy_frequency == 0:
            self.sess.run(self.assign_ops)  # Update target network
            self.sess.run(self.reset_ops)   # then reset the train network weights


class MaximumMargin(BaselineDuel):
    def __init__(self, args, environment, inputs):
        BaselineDuel.__init__(self, args, environment, inputs)

    def truth(self, train_output_states, train_output_next_states, target_output_next_states):
        return op.tofloat(self.inputs.rewards) + self.args.discount * (
        1.0 - op.tofloat(self.inputs.terminals)) * op.tofloat(op.max(target_output_next_states))

    def prediction(self, train_output_states):
        self.output = train_output_states
        return op.get(op.tofloat(train_output_states), self.inputs.actions)

    def loss(self, truth, prediction):
        _, variance = tf.nn.moments(self.output, [1])
        delta = op.optional_clip(truth - prediction, -1.0, 1.0, self.args.clip_tderror)
        return tf.to_float(tf.reduce_mean(tf.square(delta, name='square'), name='loss')) \
               + tf.reduce_mean((1.0 - variance)**2)


class OptimisticDQN(DQN):
    def __init__(self, Type, args, environment):
        DQN.__init__(self, Type, args, environment)

        # n = environment.get_num_actions()
        # qs = self.actor_output
        # qs -= op.min(qs, keep_dims=True)
        # qs /= op.max(qs, keep_dims=True)
        # approximate_one_hot_argmax = qs
        #
        # index_matrix = op.tofloat(tf.range(n)) * op.tofloat(tf.ones([args.batch_size, n]))
        # indexes = op.max(approximate_one_hot_argmax * index_matrix)
        #
        # mean_index, index_variance = tf.nn.moments(indexes, [0])
        # target_index = (n - 1.0) / 2.0
        # target_variance = target_index ** 2
        #
        # self.testop = indexes
        # self.testop2 = index_variance
        #
        # self.initialization_cost = (index_variance - target_variance)**2 # + (mean_index - target_index)**2 #

        n = environment.get_num_actions()
        indexes = range(args.batch_size)
        random.shuffle(indexes)
        target_qs = tf.one_hot(tf.constant(indexes) % n, n, on_value=3.0, off_value=2.9, axis=1)

        self.initialization_cost = tf.reduce_mean((self.actor_output - target_qs) ** 2) + tf.reduce_mean((self.train_output_states - target_qs) ** 2)

        initialization_optimizer = self.actor_network.optimizer(learning_rate=0.00001)
        self.initialization_train_op = initialization_optimizer.minimize(self.initialization_cost)
        self.initialized = False

        self.initialize_op = tf.initialize_all_variables()
        self.initialize()
        self.tensorboard()

    def train(self, **kwargs):
        data = self.build_feed_dict(**kwargs)
        if self.initialized is False:
            for i in range(10000):
                _, cost, = self.sess.run([self.initialization_train_op, self.initialization_cost], feed_dict=data)

                if cost < 0.001:
                    break

            self.initialized = True

        return DQN.train(self, **kwargs)


class WeightedLinear(Linear):
    class Inputs(Network.Inputs):
        def __init__(self, args, environment):
            Network.Inputs.__init__(self, args, environment)

            with tf.name_scope('network_specific_inputs'):
                self.lookaheads_placeholder = op.int([None, args.phi_frames] + list(environment.get_state_space()),
                                                     name='lookaheads')
                self.weights_placeholder = op.float([None], name='weight')

        def preprocessing(self):
            return {'weights': op.tofloat(self.weights_placeholder)}

    def __init__(self, args, environment, inputs):
        Network.__init__(self, args, environment, inputs)
        self.inputs = inputs

    def loss(self, truth, prediction):
        delta = op.optional_clip(truth - prediction, -1.0, 1.0, self.args.clip_tderror)
        return tf.reduce_sum(op.tofloat(self.inputs.weights_placeholder) * tf.square(delta, name='square'), name='loss')