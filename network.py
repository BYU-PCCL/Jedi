import tensorflow as tf

class Network():
    def __init__(self, args):
        self.args = args
        self.training_iterations = 0
        self.default_initializer = 'normal'

        tf.set_random_seed(args.random_seed)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.weights = []
        self.biases = []
        self.activations = []

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

    def linear(self, source, output_size, stddev=0.02, initializer='default', bias_start=0.01, activation_fn=tf.nn.relu, name='linear'):
        shape = source.get_shape().as_list()

        initializer = {
            'normal': tf.random_normal_initializer(stddev=stddev),
            'xavier': tf.contrib.layers.xavier_initializer(),
            'uniform': tf.random_uniform_initializer(),
            'truncated-normal': tf.truncated_normal_initializer(stddev=stddev)
        }[initializer if initializer != 'default' else self.default_initializer]

        with tf.variable_scope(name):
            w = tf.get_variable('matrix'+name, [shape[1], output_size], tf.float32,
                                initializer)
            b = tf.get_variable('bias'+name, [output_size],
                                initializer=tf.constant_initializer(bias_start))

            out = tf.add(tf.matmul(source, w), b)
            activated = activation_fn(out) if activation_fn is not None else out

            self.weights.append(w)
            self.biases.append(b)
            self.activations.append(activated)

            return activated, w, b

    def float(self, shape, name='float'):
        return tf.placeholder('float32', shape, name=name)

    def int(self, shape, name='int'):
        return tf.placeholder('int64', shape, name=name)

    def initialize(self):
        self.sess.run(tf.initialize_all_variables())

    def linearly_anneal(self, start, end, decay, step):
        rate = (start - end) / decay
        return tf.maximum(end, start - (rate * tf.to_float(step)))

class Linear(Network):
    def __init__(self, args, environment):
        Network.__init__(self, args)

        # Build Network
        self.state = self.float([None] + list(environment.get_state_space()), name='state')
        self.default_initializer = args.initializer

        self.s1 = self.flatten(self.state / float(environment.max_state_value()))
        self.fc1, w1, b1 = self.linear(self.s1, 500, name='fc1')
        self.fc2, w2, b2 = self.linear(self.fc1, 500, name='fc2')
        self.output, w2, b2 = self.linear(self.fc2, environment.get_num_actions(), activation_fn=None, name='output')
        self.q_action = self.argmax(self.output)
        self.q_max = self.max(self.output)

        self.next_q = self.float([None], name='next_q')
        self.action = self.int([None], name='q_action')
        self.terminal = self.float([None], name='terminal')
        self.reward = self.float([None], name='reward')

        self.action_one_hot = self.one_hot(self.action, environment.get_num_actions(), name='action')
        q_acted = self.sum(self.output * self.action_one_hot, name='q_acted')
        target_q = self.reward + args.discount * (1.0 - self.terminal) * self.next_q

        self.delta = target_q - q_acted
        self.clipped_delta = self.delta #tf.clip_by_value(self.delta, -1, 1, name='clipped_delta')

        self.loss = self.sum(self.clipped_delta**2, idx=0, name='loss')

        # Learning Rate Calculation
        self.learning_rate = self.linearly_anneal(args.learning_rate_start,
                                                  args.learning_rate_end,
                                                  args.learning_rate_decay,
                                                  self.global_step)

        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate,
                                                   decay=args.rms_decay,
                                                   momentum=float(args.rms_momentum),
                                                   epsilon=args.rms_eps).minimize(self.loss, global_step=self.global_step)

        # Initialize
        self.initialize()

    def q(self, states):
        q_action, qs = self.sess.run([self.q_action, self.output], feed_dict={self.state: states})

        return q_action, qs

    def train(self, states, actions, terminals, next_states, rewards):
        data = {
            self.state: states,
            self.action: actions,
            self.terminal: terminals,
            self.reward: rewards,
            self.next_q: self.sess.run(self.q_max, feed_dict={self.state: next_states})
        }
        _, self.training_iterations, delta, loss, lr = self.sess.run([self.optimizer, self.global_step, self.delta, self.loss, self.learning_rate], feed_dict=data)

        return delta, loss