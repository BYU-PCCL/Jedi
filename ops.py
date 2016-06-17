import tensorflow as tf
import contextlib
import copy


_context = {'floatx': tf.float32,
            'default_activation_fn': 'none'}


@contextlib.contextmanager
def context(**kwargs):
    global _context
    previous = copy.copy(_context)
    _context.update(kwargs)
    yield
    _context = previous


def one_hot(source, size, name='onehot'):
    return tf.one_hot(tf.cast(source, 'int64'), size, 1.0, 0.0, name=name)


def argmax(source, name='argmax'):
    return tf.argmax(source, dimension=1, name=name)


def max(source, name='max', keep_dims=False):
    return tf.reduce_max(source, reduction_indices=1, keep_dims=keep_dims, name=name)


def mean(source, name='mean', keep_dims=False):
    return tf.reduce_mean(source, reduction_indices=1, keep_dims=keep_dims, name=name)


def flatten(source, name='flatten'):
    shape = source.get_shape().as_list()
    dim = reduce(lambda x, y: x * y, shape[1:])

    with tf.variable_scope(name + "_flatten"):
        return tf.reshape(source, [-1, dim], name=name)


def expand(source, dim, name='expand'):
    with tf.variable_scope(name + "_expand"):
        return tf.expand_dims(source, dim)


def merge(left, right, idx=1, name='merge'):
    with tf.variable_scope(name + "_merge"):
        return tf.concat(idx, [left, right])


def sum(source, name=None, idx=1):
    return tf.reduce_sum(source, reduction_indices=idx, name=name)


def _parse_initializer(initializer, stddev):
    return {
        'normal': tf.random_normal_initializer(stddev=stddev),
        'xavier': tf.contrib.layers.xavier_initializer(),
        'uniform': tf.random_uniform_initializer(),
        'truncated-normal': tf.truncated_normal_initializer(0, stddev=stddev)
    }[initializer]


def _parse_activation(activation):
    return {
        'relu': tf.nn.relu,
        'sigmoid': tf.nn.sigmoid,
        'none': None}[activation if activation != 'default' else _context['default_activation_fn']]


def linear(source, output_size, stddev=0.02, initializer='truncated-normal', bias_start=0.01, activation_fn='default',
           name='linear'):
    shape = source.get_shape().as_list()

    initializer = _parse_initializer(initializer, stddev)
    activation_fn = _parse_activation(activation_fn)

    with tf.variable_scope(name + '_linear') as scope:
        w = tf.reshape(tf.get_variable("weight", [shape[1] * output_size], tf.float32, initializer), [shape[1], output_size])
        b = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))

        out = tf.nn.bias_add(tf.matmul(source, w), b)
        activated = activation_fn(out) if activation_fn is not None else out

        return out, activated, w, b


def conv2d(source, size, filters, stride, padding='SAME', stddev=0.02, initializer='truncated-normal', bias_start=0.01,
           activation_fn='default', name='conv2d'):
    shape = source.get_shape().as_list()
    initializer = _parse_initializer(initializer, stddev)
    activation_fn = _parse_activation(activation_fn)

    with tf.variable_scope(name + '_conv2d'):
        w = tf.reshape(tf.get_variable("weight", shape=[size * size * shape[1] * filters], initializer=initializer), [size, size, shape[1], filters])
        b = tf.get_variable("bias", [filters], initializer=tf.constant_initializer(bias_start))

        c = tf.nn.conv2d(source, w, strides=[1, 1, stride, stride], padding=padding, data_format='NCHW')
        out = tf.nn.bias_add(c, b, data_format='NCHW')
        activated = activation_fn(out) if activation_fn is not None else out

        return out, activated, w, b


def optional_clip(source, min_clip, max_clip, do):
    return tf.clip_by_value(source, min_clip, max_clip) if do else source


def get(source, index):
    return sum(source * tf.cast(one_hot(index, source.get_shape().as_list()[1]), dtype=source.dtype))


def tofloat(source, safe=True):
    if safe:
        return tf.clip_by_value(tf.cast(source, _context['floatx']), _context['floatx'].min, _context['floatx'].max)
    else:
        return tf.cast(source, _context['floatx'])


def int(shape, name='int', bits=8, unsigned=False):
    return tf.placeholder(('u' if unsigned else '') + 'int' + str(bits), shape, name=name)


def environment_scale(states, environment):
    return tf.truediv(tf.to_float(states), tf.to_float(environment.max_state_value()))