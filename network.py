import tensorflow as tf

class TrainTarget:
    def __init__(self, Type, args, environment):
        self.args = args
        self.sess = Type.create_session(args)

        self.target_network = Type(args, environment, 'target', self.sess)
        self.train_network = Type(args, environment, 'train', self.sess)

        self.training_iterations = 0
        self.lr = 0.0

        self.copy = [
            weight.assign(args.target_network_alpha * self.train_network.weights[i] + (1.0 - args.target_network_alpha) * weight, use_locking=True)
                for i, weight in enumerate(self.target_network.weights)
        ] + [
            bias.assign(args.target_network_alpha * self.train_network.biases[i] + (1.0 - args.target_network_alpha) * bias, use_locking=True)
                for i, bias in enumerate(self.target_network.biases)
        ]

    def update(self):
        self.sess.run(self.copy)

    def q(self, states):
        return self.target_network.q(states)

    def train(self, states, actions, terminals, next_states, rewards):
        response = self.train_network.train(states, actions, terminals, next_states, rewards, self.target_network)
        self.training_iterations = self.train_network.training_iterations
        self.lr = self.train_network.lr

        if self.training_iterations % self.args.copy_frequency == 0:
            self.update()

        return response


class Network():
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

        self.sess = self.create_session(args) if sess is None else sess

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        with tf.name_scope('input') as scope:
            self.state = self.float([None, args.phi_frames] + list(environment.get_state_space()), name='state') / float(environment.max_state_value())
            self.next_q = self.float([None], name='next_q')
            self.action = self.int([None], name='action_index')
            self.action_one_hot = self.one_hot(self.action, self.environment.get_num_actions(), name='one_hot_action')
            self.terminal = self.float([None], name='terminal')
            self.reward = self.float([None], name='reward')
            self.processed_reward = tf.clip_by_value(self.reward, -1.0, 1.0) if self.args.clip_reward else self.reward

        self.output = None

    def get_loss(self, q_error):
        return tf.reduce_mean(tf.square(q_error), name='loss')

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
            self.loss = self.get_loss(self.processed_delta)

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
        tf.train.SummaryWriter(self.args.tf_summary_path, self.sess.graph)

    @staticmethod
    def create_session(args):
        tf.set_random_seed(args.random_seed)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)
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

        with tf.variable_scope(name):
            return tf.reshape(source, [-1, dim], name=name)

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

    def linear(self, source, output_size, stddev=0.02, initializer='default', bias_start=0.01, activation_fn='relu', name='linear'):
        shape = source.get_shape().as_list()

        initializer = self.parse_initializer(initializer, stddev)
        activation_fn = tf.nn.relu if activation_fn == 'relu' else None

        with tf.variable_scope(name + '_linear') as scope:
            w = tf.get_variable("weight", [shape[1], output_size], tf.float32,
                                initializer)
            b = tf.get_variable("bias", [output_size],
                                initializer=tf.constant_initializer(bias_start))

            out = tf.nn.bias_add(tf.matmul(source, w), b)
            activated = activation_fn(out) if activation_fn is not None else out

            self.weights.append(w)
            self.biases.append(b)
            self.activations.append(activated)

            return activated, w, b

    def conv2d(self, source, size, filters, stride, padding='SAME', stddev=0.02, initializer='default', bias_start=0.01,
               activation_fn='relu', name='conv2d'):
        shape = source.get_shape().as_list()
        initializer = self.parse_initializer(initializer, stddev)
        activation_fn = tf.nn.relu if activation_fn == 'relu' else None

        with tf.variable_scope(name + '_conv2d') as scope:
            w = tf.get_variable("weight", shape=[size, size, shape[1], filters], initializer=initializer)
            b = tf.Variable(tf.constant(bias_start, shape=[filters]), name="bias")
            c = tf.nn.conv2d(source, w, strides=[1, 1, stride, stride], padding=padding, data_format='NCHW')
            out = tf.nn.bias_add(c, b, data_format='NCHW')
            activated = activation_fn(out) if activation_fn is not None else out

            self.weights.append(w)
            self.biases.append(b)
            self.activations.append(activated)

            return activated, w, b

    def float(self, shape, name='float'):
        return tf.placeholder('float32', shape, name=name)

    def int(self, shape, name='int'):
        return tf.placeholder('int64', shape, name=name)

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
        with tf.variable_scope(name) as scope:
            Network.__init__(self, args, environment, name, sess)

            # Build Network
            self.conv1,  w1, b1 = self.conv2d(self.state, size=8, filters=32, stride=4, name='conv1')
            self.conv2,  w2, b2 = self.conv2d(self.conv1, size=4, filters=64, stride=2, name='conv2')
            self.conv3,  w3, b3 = self.conv2d(self.conv2, size=3, filters=64, stride=1, name='conv3')
            self.fc4,    w4, b4 = self.linear(self.flatten(self.conv3), 512, name='fc4')
            self.output, w5, b5 = self.linear(self.fc4, environment.get_num_actions(), activation_fn='none', name='output')

            self.post_init()


class Linear(Network):
    def __init__(self, args, environment, name='linear_network', sess=None):
        with tf.variable_scope(name) as scope:
            Network.__init__(self, args, environment, name, sess)

            self.fc1,    w1, b1 = self.linear(self.flatten(self.state), 500, name='fc1')
            self.fc2,    w2, b2 = self.linear(self.fc1, 500, name='fc2')
            self.output, w2, b2 = self.linear(self.fc2, environment.get_num_actions(), activation_fn='none', name='output')

            self.post_init()

class Constrained(Network):
    def __init__(self, args, environment, name='linear_network', sess=None):
        with tf.variable_scope(name) as scope:
            Network.__init__(self, args, environment, name, sess)

            # Build Network
            self.conv1,  w1, b1 = self.conv2d(self.state, size=8, filters=32, stride=4, name='conv1')
            self.conv2,  w2, b2 = self.conv2d(self.conv1, size=4, filters=64, stride=2, name='conv2')
            self.conv3,  w3, b3 = self.conv2d(self.conv2, size=3, filters=64, stride=1, name='conv3')
            self.fc4,    w4, b4 = self.linear(self.flatten(self.conv3), 512, name='fc4')
            self.output, w5, b5 = self.linear(self.fc4, environment.get_num_actions(), activation_fn='none', name='output')

            self.post_init()

    def get_loss(self, q_error):
        return tf.reduce_mean(tf.square(q_error), name='loss')