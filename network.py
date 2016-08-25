import tensorflow as tf
import subprocess
import ops as op
from functools import reduce
import os

class Network(object):
    class Inputs:
        def __getattr__(self, item):
            processed = {'actions': self.actions_placeholder,
                         'rewards': op.optional_clip(self.rewards_placeholder, -1.0, 1.0, self.args.clip_reward),
                         'states': op.environment_scale(self.states_placeholder, self.environment),
                         'next_states': op.environment_scale(self.next_states_placeholder, self.environment),
                         'terminals': self.terminals_placeholder}

            processed.update(self.preprocessing())

            return processed[item]

        def __init__(self, args, environment):
            self.args = args
            self.environment = environment

            with tf.name_scope('inputs'):
                self.states_placeholder = op.int([None, args.phi_frames] + list(environment.get_state_space()),
                                                 name='state', unsigned=True)
                self.next_states_placeholder = op.int([None, args.phi_frames] + list(environment.get_state_space()),
                                                      name='next_state', unsigned=True)
                self.actions_placeholder = op.float([None] + list(environment.get_action_space()), name='actions')
                self.terminals_placeholder = op.float([None], name='terminal')
                self.rewards_placeholder = op.float([None], name='reward')

        def preprocessing(self):
            return {}

    def __init__(self, args, environment):
        self.args = args
        self.environment = environment
        self.tensorboard_process = None
        self.training_iterations = 0
        self.batch_loss = 0
        self.learning_rate = 0

        self.sess = self.start_session(args)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.learning_rate_op = self.build_learning_rate(step=self.global_step)

        with op.context(floatx=tf.float32, floatsafe=False):

            self.train_op = None
            self.priority_op = None
            self.loss_op = None
            self.agent_output_action = None
            self.agent_output = None
            self.agent_network = None

            self.build_networks()

            assert (self.train_op is not None and
                    self.priority_op is not None and
                    self.loss_op is not None and
                    self.agent_output_action is not None and
                    self.agent_output is not None), 'Network implementation must define the operations found on this line'

            self.assign_ops = self.build_assign_ops()

        self.initialize()

    def build(self, states):
        pass

    def truth(self, train_output_states, train_output_next_states, target_output_next_states):
        return op.tofloat(self.inputs.rewards) + self.args.discount * (
        1.0 - op.tofloat(self.inputs.terminals)) * op.tofloat(op.max(target_output_next_states))

    def prediction(self, train_output_states):
        return op.get(op.tofloat(train_output_states), self.inputs.actions)

    def loss(self, truth, prediction):
        assert prediction.get_shape().as_list() == truth.get_shape().as_list(), 'prediction and truth shapes must match'
        delta = op.optional_clip(truth - prediction, -1.0, 1.0, self.args.clip_tderror)
        return tf.reduce_mean(tf.square(delta, name='square'), name='loss')

    def priority(self, truth, prediction):
        return tf.pow(truth - prediction, self.args.priority_temperature)

    def action(self, output):
        return op.argmax(output)

    def build_networks(self):

        self.inputs = self.Inputs(self.args, self.environment)

        with tf.device('/gpu:0'):
            with tf.variable_scope('target_network'):
                target_output_next_states = self.build(self.inputs.next_states)

        with tf.device('/gpu:1'):
            with tf.variable_scope('train_network'):
                train_output_states = self.build(self.inputs.states)

            with tf.variable_scope('train_network', reuse=True):
                train_output_next_states = self.build(self.inputs.next_states)

        with tf.device('/gpu:0'):
            with tf.name_scope('thread_actor'), tf.variable_scope('target_network', reuse=True):
                self.agent_output = self.build(self.inputs.states)
                self.agent_output_action = self.action(self.agent_output)

        with tf.device('/gpu:1'):
            with tf.name_scope('loss'):
                truth = self.truth(train_output_states=train_output_states,
                                   train_output_next_states=train_output_next_states,
                                   target_output_next_states=target_output_next_states)

                truth = tf.stop_gradient(truth)

                prediction = self.prediction(train_output_states=train_output_states)

                self.loss_op = self.loss(truth=truth, prediction=prediction)
                self.priority_op = self.priority(truth=truth, prediction=prediction)

        with tf.name_scope('optimizer'):
            self.train_op = self.build_train_op(learning_rate=self.learning_rate_op,
                                                loss=self.loss_op,
                                                global_step=self.global_step)

    def build_train_op(self, learning_rate, loss, global_step):
        # It's about 2x faster for us to compute/apply the gradients than to use optimizer.minimize()
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                              decay=self.args.rms_decay,
                                              momentum=float(self.args.rms_momentum),
                                              epsilon=self.args.rms_eps)
        gradient = optimizer.compute_gradients(loss, colocate_gradients_with_ops=True)
        gradients = [(grad, var) for grad, var in gradient if grad is not None]

        return optimizer.apply_gradients(gradients, global_step=global_step)

    def build_assign_ops(self):
        with tf.name_scope('update_ops'):
            target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_network')
            train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='train_network')

            return [
                # https://www.tensorflow.org/versions/r0.10/api_docs/python/train.html#ExponentialMovingAverage
                target.assign_sub((1 - self.args.target_network_alpha) * (target - train)) for
                target, train in zip(target_vars, train_vars)]

    def initialize(self):
        self.sess.run(tf.initialize_all_variables())
        self.tensorboard()

    def total_parameters(self):
        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='train_network')
        return sum([sum([reduce(lambda x, y: x * y, l.get_shape().as_list()) for l in e]) for e in [vars]])

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
                                                    stdout=None if self.args.verbose else open(os.devnull, 'w'),
                                                    stderr=None if self.args.verbose else open(os.devnull, 'w'),
                                                    close_fds=True)

    def update(self):
        if self.training_iterations % self.args.copy_frequency == 0:
            self.sess.run(self.assign_ops)

    def build_feed_dict(self, **kwargs):
        ignored = [key for key in kwargs.keys() if key + '_placeholder' not in self.inputs.__dict__.keys()]

        # Optional print a notice to the user if they are passing in unknown variables
        # assert len(ignored) == 0, 'The following arguments passed to train() are not used by this network : ' + str(ignored)

        return {getattr(self.inputs, key + '_placeholder'): var for (key, var) in kwargs.items() if key not in ignored}

    def train(self, **kwargs):
        data = self.build_feed_dict(**kwargs)

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

        agent_output = self.agent_output
        additional_ops = []

        if type(self.agent_output) is list:
            agent_output = self.agent_output[0]
            additional_ops = self.agent_output[1:]

        results = self.sess.run([self.agent_output_action, agent_output] + additional_ops,
                                feed_dict=data)

        return results[0], results[1], results[2:]

    def build_learning_rate(self, step):
        decayed_lr = tf.train.exponential_decay(self.args.learning_rate_start,
                                                step,
                                                self.args.learning_rate_decay_step,
                                                self.args.learning_rate_decay,
                                                staircase=False)
        return tf.maximum(self.args.learning_rate_end, decayed_lr)


class Linear(Network):
    def build(self, states):
        with op.context(default_activation_fn='relu'):
            fc1,    w1, b1 = op.linear(op.flatten(states, name="fc1_flatten"), 500, name='fc1')
            fc2,    w2, b2 = op.linear(fc1, 500, name='fc2')
            value,  w3, b3 = op.linear(fc2, self.environment.get_num_actions(), activation_fn='none', name='value')
            advantages, w4, b4 = op.linear(fc2, self.environment.get_num_actions(), activation_fn='none', name='advantages')

            # Dueling DQN - http://arxiv.org/pdf/1511.06581v3.pdf
            output = value + (advantages - op.mean(advantages, keep_dims=True))

        return output


class Baseline(Network):
    def build(self, states):
        with op.context(default_activation_fn='relu'):
            conv1, w1, b1 = op.conv2d(states, size=8, filters=32, stride=4, name='conv1')
            conv2, w2, b2 = op.conv2d(conv1, size=4, filters=64, stride=2, name='conv2')
            conv3, w3, b3 = op.conv2d(conv2, size=3, filters=64, stride=1, name='conv3')
            fc4, w4, b4 = op.linear(op.flatten(conv3, name="fc4"), 512, name='fc4')
            output, w5, b5 = op.linear(fc4, self.environment.get_num_actions(), activation_fn='none', name='output')

            return output


class BaselineDuel(Network):
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


class BaselineDouble(Network):
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
    def build(self, states):
        with op.context(default_activation_fn='relu'):
            conv1,    w1, b1 = op.conv2d(states, size=8, filters=32, stride=4, name='conv1')
            conv2,    w2, b2 = op.conv2d(conv1, size=4, filters=64, stride=2, name='conv2')
            conv3,    w3, b3 = op.conv2d(conv2, size=3, filters=64, stride=1, name='conv3')
            fc4,      w4, b4 = op.linear(op.flatten(conv3, name="fc4"), 512, name='fc4')
            output,   w5, b5 = op.linear(fc4, self.environment.get_num_actions(), activation_fn='none', name='output')
            raw_sigma, w6, b6 = op.linear(fc4, self.environment.get_num_actions(), name='variance')

            raw_sigma += 0.0001  # to avoid divide by zero
            sigma = tf.exp(raw_sigma)

        return output, sigma

    def truth(self, train_output_states, train_output_next_states, target_output_next_states):
        return op.tofloat(self.inputs.rewards) + self.args.discount * (
        1.0 - op.tofloat(self.inputs.terminals)) * op.tofloat(op.max(target_output_next_states[0]))

    def prediction(self, train_output_states):
        sigma = train_output_states[0]
        return op.get(op.tofloat(train_output_states), self.inputs.actions), sigma

    def loss(self, truth, prediction):
        y = prediction[0]
        mu = truth
        sigma = op.get(prediction[1], self.inputs.actions)

        # Gaussian log-likelihood
        result = op.tofloat(y - mu)  # Primarily to prevent under/overflow since they are already float16
        result = tf.cast(result, 'float32') * tf.inv(sigma)
        result = -tf.square(result) / 2
        result = result + tf.log(tf.inv(sigma))

        return tf.reduce_mean(-result)


class Causal(Network):
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


class MaximumMargin(BaselineDuel):
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

    def loss(self, truth, prediction):
        delta = op.optional_clip(truth - prediction, -1.0, 1.0, self.args.clip_tderror)
        weights = op.optional_clip(self.inputs.weights_placeholder, 0, 100, True)
        return tf.reduce_mean(weights * tf.square(delta, name='square'), name='loss')


class ActorCritic(Network):
    class Inputs(Network.Inputs):
        def __init__(self, args, environment):
            Network.Inputs.__init__(self, args, environment)

            with tf.name_scope('inputs'):
                self.states_placeholder = op.float([None, args.phi_frames] + list(environment.get_state_space()), name='state')
                self.next_states_placeholder = op.float([None, args.phi_frames] + list(environment.get_state_space()), name='next_state')

    def build_networks(self):
        self.inputs = self.Inputs(self.args, self.environment)

        with tf.device('/gpu:0'):
            with tf.variable_scope('target_network'):
                target_actor = self.actor(self.inputs.next_states_placeholder)
                target_critic_next_states = self.critic(self.inputs.next_states_placeholder, target_actor)

            with tf.name_scope('train_actor_critic'), tf.variable_scope('train_network'):
                self.train_actor_states = self.actor(self.inputs.states_placeholder)
                self.train_critic_states_actor_actions = self.critic(self.inputs.states_placeholder, self.train_actor_states)

            with tf.name_scope('train_critic_action_placeholder'), tf.variable_scope('train_network', reuse=True):
                train_critic_states_placeholder_actions = self.critic(self.inputs.states_placeholder, self.inputs.actions_placeholder)

        with tf.device('/gpu:1'):
            with tf.name_scope('thread_actor'), tf.variable_scope('target_network', reuse=True):
                self.agent_output_action = self.actor(self.inputs.states_placeholder)
                self.agent_output = self.critic(self.inputs.states_placeholder, self.agent_output_action)

            with tf.name_scope('loss'):
                truth = self.inputs.rewards + self.args.discount * (1.0 - self.inputs.terminals) * op.max(target_critic_next_states)
                prediction = op.max(train_critic_states_placeholder_actions)

                truth = tf.stop_gradient(truth)

                self.loss_op = tf.reduce_mean(tf.square(truth - prediction)) + tf.add_n([0.01 * tf.nn.l2_loss(var) for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='train_network/critic')])
                self.priority_op = tf.pow(truth - prediction, self.args.priority_temperature)

            with tf.name_scope('optimizer'):
                self.train_op = self.build_train_op()

    def actor(self, state):
        with tf.variable_scope('actor'):
            fc1, w1, b1 = op.linear(op.flatten(state), 400, name='fc1', stddev=0.001, bias_start=0.001)
            fc2, w2, b2 = op.linear(fc1, 300, name='fc2', stddev=0.001, bias_start=0.001)
            action, w3, b3 = op.linear(fc2, self.environment.get_num_actions(), name='actions', activation_fn='tanh', stddev=0.001, bias_start=0.001)

            return action * 2

    def critic(self, state, action):
        with tf.variable_scope('critic'):
            fc1, w1, b1 = op.linear(op.flatten(state), 400, name='fc1', stddev=0.001, bias_start=0.001)
            fc2, w2, b2 = op.linear(fc1, 300, name='fc2', activation_fn='none', stddev=0.001, bias_start=0.001)
            fc2a, w3, b3 = op.linear(action, 300, name='fc3', activation_fn='none', stddev=0.001, bias_start=0.001)
            q, w4, b4 = op.linear(tf.nn.relu(fc2 + fc2a - b2), 1, name='value', activation_fn='none', stddev=0.001, bias_start=0.001)

            return q

    def build_train_op(self):
        actor_optimizer = tf.train.AdamOptimizer(0.0001)
        critic_optimizer = tf.train.AdamOptimizer(0.001)

        critic_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='train_network/critic')
        actor_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='train_network/actor')

        update_critic = critic_optimizer.minimize(self.loss_op, var_list=critic_variables)

        with tf.control_dependencies([update_critic]):
            # grad_action(q_value)
            action_gradient = tf.gradients(self.train_critic_states_actor_actions, [self.train_actor_states])[0]

            # grad_theta_actor(actions)
            actor_gradients = tf.gradients(self.train_actor_states, actor_variables, -action_gradient)

            actor_gradients = zip(actor_gradients, actor_variables)

            return actor_optimizer.apply_gradients(actor_gradients, global_step=self.global_step)