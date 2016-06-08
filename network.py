import tensorflow as tf
import subprocess
import os
import numpy as np
import ops as op

class QLearner(object):
    def __init__(self, Type, args, environment):
        self.args = args
        self.environment = environment

        self.training_iterations = 0
        self.batch_loss = 0
        self.learning_rate = 0
        gradients = []

        self.sess = self.start_session(args)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        inputs = Type.Inputs(args, environment)

        with tf.device('/gpu:0'):
            with tf.variable_scope('target_network'):
                target_network = Type(args, environment, inputs)
                target_output = target_network.build(inputs.next_states)

            with tf.variable_scope('train_network'):
                train_network = Type(args, environment, inputs)
                train_output = train_network.build(inputs.states)

            with tf.name_scope('thread_actor'), tf.variable_scope('target_network', reuse=True):
                self.actor_network = Type(args, environment, inputs)
                self.actor_output = self.actor_network.build(inputs.states)
                self.actor_output_argmax = op.argmax(self.actor_output)

        with tf.device('/gpu:1'):
            with tf.name_scope('loss'):
                truth = train_network.truth(train_output, target_output)
                prediction = tf.stop_gradient(train_network.prediction(train_output, target_output))

                self.loss_op = train_network.loss(truth=truth, prediction=prediction)
                self.priority_op = train_network.priority(truth=truth, prediction=prediction)

            # It's about 2x faster for us to compute the gradients than to use optimizer.minimize()
            with tf.name_scope('optimizer'):
                self.learning_rate_op = train_network.learning_rate(step=self.global_step)
                optimizer = train_network.optimizer(self.learning_rate_op)
                gradient = optimizer.compute_gradients(self.loss_op)
                gradients += [(grad, var) for grad, var in gradient if grad is not None]
                self.train_op = optimizer.apply_gradients(gradients, global_step=self.global_step)

            self.target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_network')
            self.train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='train_network')
            self.assign_ops = [target.assign(train) for target, train in zip(self.target_vars, self.train_vars)]

            self.inputs = inputs

            self.test = self.loss_op

        self.sess.run(tf.initialize_all_variables())
        self.tensorboard()

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
        tf.train.SummaryWriter(self.args.tf_summary_path, self.sess.graph)
        self.tensorboard_process = subprocess.Popen(["tensorboard", "--logdir=" + self.args.tf_summary_path],
                                                    stdout=open(os.devnull, 'w'),
                                                    stderr=open(os.devnull, 'w'),
                                                    close_fds=True)


    def update(self):
        self.sess.run(self.assign_ops)

    def build_feed_dict(self, **kwargs):
        # Print a notice to the user if they are passing in unknown variables
        ignored = [key for key in kwargs.keys() if key + '_placeholder' not in self.inputs.__dict__.keys()]
        assert len(ignored) == 0, 'Arguments passed to train() are ignored : ' + str(ignored)
        return {getattr(self.inputs, key + '_placeholder'): var for (key, var) in kwargs.iteritems()}

    def train(self, **kwargs):
        data = self.build_feed_dict(**kwargs)

        _, priority, self.batch_loss, self.learning_rate, self.training_iterations, t = self.sess.run([self.train_op,
                                                                                                    self.priority_op,
                                                                                                    self.loss_op,
                                                                                                    self.learning_rate_op,
                                                                                                    self.global_step, self.test],
                                                                                                   feed_dict=data)


        print t

        if self.training_iterations % self.args.copy_frequency == 0:
            self.update()

        return priority, self.batch_loss

    def q(self, **kwargs):
        data = self.build_feed_dict(**kwargs)
        results = self.sess.run([self.actor_output_argmax, self.actor_output] + self.actor_network.additional_q_ops,
                                feed_dict=data)

        return results[0], results[1], results[2:]


class Network(object):

    class Inputs:
        def __getattr__(self, item):
            processed = {'actions': self.actions_placeholder,
                         'rewards': op.optional_clip(self.rewards_placeholder, -1, 1, self.args.clip_reward),
                         'states': op.environment_scale(self.states_placeholder, self.environment),
                         'next_states': op.environment_scale(self.states_placeholder, self.environment),
                         'terminals': self.terminals_placeholder}

            processed.update(self.additional_inputs())

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

        def additional_inputs(self):
            return {}

    def __init__(self, args, environment, inputs):
        self.args = args
        self.environment = environment
        self.inputs = inputs
        self.additional_q_ops = []

    def build(self, states):
        pass

    def truth(self, train_output, target_output):
        return op.float16(self.inputs.rewards) + op.float16(self.args.discount) * (op.float16(1.0) - op.float16(self.inputs.terminals)) * op.float16(op.max(target_output))

    def prediction(self, train_output, target_output):
        return op.get(op.float16(train_output), self.inputs.actions, self.environment.get_num_actions())

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

    def loss(self, truth, prediction):
        delta = op.optional_clip(truth - prediction, -1.0, 1.0, self.args.clip_tderror)
        return tf.reduce_mean(tf.square(delta, name='square'), name='loss')


class Baseline(Network):
    def __init__(self, args, environment, inputs):
        Network.__init__(self, args, environment, inputs)

    def build(self, states):
        conv1, w1, b1 = op.conv2d(states, size=8, filters=32, stride=4, name='conv1')
        conv2, w2, b2 = op.conv2d(conv1, size=4, filters=64, stride=2, name='conv2')
        conv3, w3, b3 = op.conv2d(conv2, size=3, filters=64, stride=1, name='conv3')
        fc4, w4, b4 = op.linear(op.flatten(conv3, name="fc4"), 512, name='fc4')
        output, w5, b5 = op.linear(fc4, self.environment.get_num_actions(), activation_fn='none', name='output')

        return output


class Linear(Network):
    def __init__(self, args, environment, inputs):
        Network.__init__(self, args, environment, inputs)

    def build(self, states):
        fc1,    w1, b1 = op.linear(op.flatten(states, name="fc1"), 500, name='fc1')
        fc2,    w2, b2 = op.linear(fc1, 500, name='fc2')
        output, w2, b2 = op.linear(fc2, self.environment.get_num_actions(), activation_fn='none', name='output')

        return output


class Constrained(Network):

    class Inputs(Network.Inputs):
        def __init__(self, args, environment):
            Network.Inputs.__init__(self, args, environment)

            with tf.name_scope('network_specific_inputs'):
                self.lookaheads_placeholder = op.int([None, args.phi_frames] + list(environment.get_state_space()),
                                                    name='lookaheads')

        def additional_inputs(self):
            return {'lookaheads': op.environment_scale(self.states_placeholder, self.environment)}

    def __init__(self, args, environment, inputs):
        Network.__init__(self, args, environment, inputs)

    def build(self, states):

        with tf.variable_scope('net'):
            conv1,     w1, b1 = op.conv2d(states, size=8, filters=32, stride=4, name='conv1')
            conv2,     w2, b2 = op.conv2d(conv1, size=4, filters=64, stride=2, name='conv2')
            conv3,     w3, b3 = op.conv2d(conv2, size=3, filters=64, stride=1, name='conv3')
            fc4,       w4, b4 = op.linear(op.flatten(conv3), 256, name='fc4')

            h,         w5, b5 = op.linear(fc4, 256, name='h')
            h1,        w6, b6 = op.linear(h, 256, name='h1')
            hhat,      w7, b7 = op.linear(h1, 256, name='hhat')

            fc8,       w8, b8 = op.linear(op.merge(h, hhat, name="fc8"), 256, name='fc8')
            output,    w9, b9 = op.linear(fc8, self.environment.get_num_actions(), activation_fn='none', name='output')

        with tf.name_scope('prediction'), tf.variable_scope('net', reuse=True):
            hhat_conv1, _, _ = op.conv2d(self.inputs.lookaheads, size=8, filters=32, stride=4, name='conv1')
            hhat_conv2, _, _ = op.conv2d(hhat_conv1, size=4, filters=64, stride=2, name='conv2')
            hhat_conv3, _, _ = op.conv2d(hhat_conv2, size=3, filters=64, stride=1, name='conv3')
            hhat_truth, _, _ = op.linear(op.flatten(hhat_conv3), 256, name='fc4')
            self.constraint_error = tf.reduce_mean((hhat - hhat_truth)**2, reduction_indices=1, name='prediction_error')

        return output

    def loss(self, truth, prediction):
        delta = op.optional_clip(truth - prediction, -1.0, 1.0, self.args.clip_tderror)
        return tf.reduce_mean(tf.square(delta)) + tf.reduce_mean(op.float16(self.constraint_error))


class Density(Network):
    def __init__(self, args, environment, inputs):
        Network.__init__(self, args, environment, inputs)

    def build(self, states):
        conv1,    w1, b1 = op.conv2d(states, size=8, filters=32, stride=4, name='conv1')
        conv2,    w2, b2 = op.conv2d(conv1, size=4, filters=64, stride=2, name='conv2')
        conv3,    w3, b3 = op.conv2d(conv2, size=3, filters=64, stride=1, name='conv3')
        fc4,      w4, b4 = op.linear(op.flatten(conv3, name="fc4"), 512, name='fc4')
        output,   w5, b5 = op.linear(fc4, self.environment.get_num_actions(), activation_fn='none', name='output')
        self.variance, w6, b6 = op.linear(fc4, self.environment.get_num_actions(), activation_fn='sigmoid', name='variance')

        self.additional_q_ops.append(self.variance)

        return output

    def loss(self, truth, prediction):

        y = prediction
        mu = truth
        sigma = self.sigma

        out_sigma
        out_mu, y

        result = tf.sub(y, mu)
        result = tf.mul(result, tf.inv(sigma))
        result = -tf.square(result) / 2
        result = tf.mul(tf.exp(result), tf.inv(sigma)) * (1 / np.sqrt(2*np.pi))
        result = tf.reduce_sum(result, 1, keep_dims=True)
        result = -tf.log(result)
        return tf.reduce_mean(result)

        sigmas = self.sum(self.float16(self.variance) * self.float16(qlearner.actions), name='variance_acted') + self.float16(1)
        self.sigma = tf.reduce_mean(sigmas)

        return tf.log(self.sigma) + tf.square(tf.reduce_mean(processed_delta)) / (2.0 * tf.square(self.sigma))


class Causal(Network):
    def __init__(self, args, environment, inputs):
        Network.__init__(self, args, environment, inputs)

    def build(self, states):

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


class Convergence(Network):
    def __init__(self, args, environment, inputs):
        Network.__init__(self, args, environment, inputs)
        assert args.agent_type == 'convergence', 'Convergence Commander must use Convergence Agent'

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
