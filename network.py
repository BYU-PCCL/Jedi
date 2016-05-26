import tensorflow as tf

class TrainTarget:
    def __init__(self, Type, args, environment):

        self.args = args

        self.sess = Type.create_session(args)

        self.target_network = Type(args, environment, 'target', self.sess)
        self.train_network = Type(args, environment, 'train', self.sess)
        self.training_iterations = 0

        self.copy = [
            weight.assign(self.train_network.weights[i]) for i, weight in enumerate(self.target_network.weights)
        ] + [
            bias.assign(self.train_network.biases[i]) for i, bias in enumerate(self.target_network.biases)
        ]

    def update(self):
        self.sess.run(self.copy)

    def q(self, states):
        return self.target_network.q(states)

    def train(self, states, actions, terminals, next_states, rewards):

        self.training_iterations += 1

        if self.training_iterations % self.args.copy_frequency == 0:
            self.update()

        return self.train_network.train(states, actions, terminals, next_states, rewards, self.target_network)

class Network():
    def __init__(self, args, environment, name='network', sess=None):
        self.args = args
        self.environment = environment
        self.name = name
        self.training_iterations = 0
        self.batch_loss = 0
        self.lr = 0
        self.default_initializer = 'normal'

        self.sess = self.create_session(args) if sess is None else sess

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.weights = []
        self.biases = []
        self.activations = []

        self.state = self.float([None, args.phi_frames] + list(environment.get_state_space()), name=self.name + '_state') / float(environment.max_state_value())
        self.default_initializer = args.initializer

    def post_init(self):
        self.next_q = self.float([None], name=self.name + '_next_q')
        self.action = self.int([None], name=self.name + '_q_action')
        self.terminal = self.float([None], name=self.name + '_terminal')
        self.reward = self.float([None], name=self.name + '_reward')
        self.q_action = self.argmax(self.output)
        self.q_max = self.max(self.output)

        self.action_one_hot = self.one_hot(self.action, self.environment.get_num_actions(), name=self.name + '_action')
        q_acted = self.sum(self.output * self.action_one_hot, name=self.name + '_q_acted')
        target_q = self.reward + self.args.discount * (1.0 - self.terminal) * self.next_q

        self.delta = target_q - q_acted
        self.clipped_delta = self.delta  # tf.clip_by_value(self.delta, -1, 1, name='clipped_delta')

        self.loss = self.sum(self.clipped_delta ** 2, idx=0, name=self.name + '_oss')

        # Learning Rate Calculation
        self.learning_rate = self.linearly_anneal(self.args.learning_rate_start,
                                                  self.args.learning_rate_end,
                                                  self.args.learning_rate_decay,
                                                  self.global_step)

        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate,
                                                   decay=self.args.rms_decay,
                                                   momentum=float(self.args.rms_momentum),
                                                   epsilon=self.args.rms_eps).minimize(self.loss, global_step=self.global_step)

        # Initialize
        self.initialize()

    @staticmethod
    def create_session(args):
        tf.set_random_seed(args.random_seed)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    def one_hot(self, source, size, name='onehot'):
        return tf.one_hot(source, size, 1.0, 0.0, name=self.name + '_' + name)

    def argmax(self, source, name='argmax'):
        return tf.argmax(source, dimension=1)

    def max(self, source, name='max'):
        return tf.reduce_max(source, reduction_indices=1, name=self.name + '_' + name)

    def flatten(self, source, name='flatten'):
        shape = source.get_shape().as_list()
        dim = reduce(lambda x, y: x*y, shape[1:])

        with tf.variable_scope(name):
            return tf.reshape(source, [-1, dim], name=self.name + '_' + name)

    def sum(self, source, name, idx=1):
        return tf.reduce_sum(source, reduction_indices=idx, name=self.name + '_' + name)

    def squared_sum(self, source, name):
        return tf.reduce_sum(tf.square(source), reduction_indices=0, name=self.name + '_' + name)

    def parse_initializer(self, initializer, stddev):
        return {
            'normal': tf.random_normal_initializer(stddev=stddev),
            'xavier': tf.contrib.layers.xavier_initializer(),
            'uniform': tf.random_uniform_initializer(),
            'truncated-normal': tf.truncated_normal_initializer(stddev=stddev)
        }[initializer if initializer != 'default' else self.default_initializer]

    def linear(self, source, output_size, stddev=0.02, initializer='default', bias_start=0.01, activation_fn=tf.nn.relu, name='linear'):
        shape = source.get_shape().as_list()

        initializer = self.parse_initializer(initializer, stddev)

        with tf.variable_scope(name):
            w = tf.get_variable(self.name + '_matrix'+name, [shape[1], output_size], tf.float32,
                                initializer)
            b = tf.get_variable(self.name + '_bias'+name, [output_size],
                                initializer=tf.constant_initializer(bias_start))

            out = tf.nn.bias_add(tf.matmul(source, w), b)
            activated = activation_fn(out) if activation_fn is not None else out

            self.weights.append(w)
            self.biases.append(b)
            self.activations.append(activated)

            return activated, w, b

    def conv2d(self, source, size, filters, stride, padding='SAME', stddev=0.02, initializer='default', bias_start=0.01, activation_fn=tf.nn.relu, name='conv2d'):
        shape = source.get_shape().as_list()
        initializer = self.parse_initializer(initializer, stddev)

        w = tf.get_variable(self.name + '_weight_' + name, shape=[size, size, shape[1], filters], initializer=initializer)
        b = tf.Variable(tf.constant(bias_start, shape=[filters]), name=self.name + '_bias_' + name)
        c = tf.nn.conv2d(source, w, strides=[1, 1, stride, stride], padding=padding, name=name, data_format='NCHW')
        out = tf.nn.bias_add(c, b, data_format='NCHW')
        activated = activation_fn(out) if activation_fn is not None else out

        self.weights.append(w)
        self.biases.append(b)
        self.activations.append(activated)

        return activated, w, b

    def float(self, shape, name='float'):
        return tf.placeholder('float32', shape, name=self.name + '_' + name)

    def int(self, shape, name='int'):
        return tf.placeholder('int64', shape, name=self.name + '_' + name)

    def initialize(self):
        self.sess.run(tf.initialize_all_variables())

    def linearly_anneal(self, start, end, decay, step):
        rate = (start - end) / decay
        return tf.maximum(end, start - (rate * tf.to_float(step)))

    def q(self, states):
        q_action, qs = self.sess.run([self.q_action, self.output], feed_dict={self.state: states})

        return q_action, qs


    def train(self, states, actions, terminals, next_states, rewards, target_network=None):

        network = self if target_network is None else target_network

        data = {
            self.state: states,
            self.action: actions,
            self.terminal: terminals,
            self.reward: rewards,
            self.next_q: self.sess.run(network.q_max, feed_dict={network.state: next_states})
        }
        _, self.training_iterations, delta, self.batch_loss, self.lr = self.sess.run(
            [self.optimizer, self.global_step, self.delta, self.loss, self.learning_rate], feed_dict=data)

        return delta, self.batch_loss


class Baseline(Network):
    def __init__(self, args, environment, name='baseline_network', sess=None):
        Network.__init__(self, args, environment, name, sess)

        # Build Network
        self.conv1,  w1, b1 = self.conv2d(self.state, size=8, filters=16, stride=4, name='conv1')
        self.conv2,  w2, b2 = self.conv2d(self.conv1, size=4, filters=32, stride=2, name='conv2')
        self.fc3,    w3, b3 = self.linear(self.flatten(self.conv2), 256, name='fc3')
        self.output, w4, b4 = self.linear(self.fc3, environment.get_num_actions(), activation_fn=None, name='output')

        self.post_init()


class Linear(Network):
    def __init__(self, args, environment, name='linear_network', sess=None):
        Network.__init__(self, args, environment, name, sess)

        self.fc1,    w1, b1 = self.linear(self.flatten(self.state), 500, name='fc1')
        self.fc2,    w2, b2 = self.linear(self.fc1, 500, name='fc2')
        self.output, w2, b2 = self.linear(self.fc2, environment.get_num_actions(), activation_fn=None, name='output')

        self.post_init()