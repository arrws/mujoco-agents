
import numpy as np
import tensorflow as tf
from gym.spaces import Box, Discrete
import scipy.signal


# NETWORK HELPERS

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=combined_shape(None,dim))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

def placeholder_from_space(space):
    if isinstance(space, Box):
        return placeholder(space.shape)
    elif isinstance(space, Discrete):
        return tf.placeholder(dtype=tf.int32, shape=(None,))
    raise NotImplementedError

def placeholders_from_spaces(*args):
    return [placeholder_from_space(space) for space in args]


def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    # multi-layered-perceptron
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

def get_policy(action_space):
    # returns discreete or continuous policy function
    policy = None
    if isinstance(action_space, Box):
        policy = mlp_gaussian_policy
    elif isinstance(action_space, Discrete):
        policy = mlp_categorical_policy
    return policy

def mlp_categorical_policy(x, a, hidden_sizes, activation, output_activation, action_space):
    act_dim = action_space.n
    logits = mlp(x, list(hidden_sizes)+[act_dim], activation, None)
    logp_all = tf.nn.log_softmax(logits)
    pi = tf.squeeze(tf.multinomial(logits,1), axis=1)
    logp = tf.reduce_sum(tf.one_hot(a, depth=act_dim) * logp_all, axis=1)
    logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1)
    return pi, logp, logp_pi

def mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation, action_space):
    act_dim = a.shape.as_list()[-1]
    mu = mlp(x, list(hidden_sizes)+[act_dim], activation, output_activation)
    log_std = tf.get_variable(name='log_std', initializer=-0.5*np.ones(act_dim, dtype=np.float32))
    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    logp = gaussian_likelihood(a, mu, log_std)
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    return pi, logp, logp_pi




# Logger Class for printing

def get_stats(x):
    # computers mean, standard deviation, min and max of an array
    mean = np.sum(x) / len(x)
    std = np.sqrt(np.sum(x-mean)**2 / len(x))
    return [mean, std, np.max(x), np.min(x)]

class Logger:
    def __init__(self):
        self.data = dict()

    def store(self, **kwargs):
        for k,v in kwargs.items():
            if not(k in self.data.keys()):
                self.data[k] = []
            self.data[k].append(v)

    def log(self, key, val=None, with_min_and_max=False, average_only=False):
        if val is not None:
            print(key,'\t',val)
        else:
            stats = get_stats(self.data[key])
            print(key + '\tAvg\t', stats[0])
            if not(average_only):
                print('\tStd\t', stats[1])
            if with_min_and_max:
                print('\tMn/Mx\t', stats[3], '\t', stats[2])
        self.data[key] = []



# Buffer Class for storing trajectories experience

class Buffer:
    # uses Generalized Advantage Estimation (GAE-Lambda) for calculating the advantages of state-action pairs.
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)

        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)

        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)

        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        # aprox with GAE-Lambda the  advantage and rewards-to-go to use as targets in case of tracjectory cut off
        # last_val = 0 if agent reached done, otherwise V(s_T)

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = self.discount_cumsum(deltas, self.gamma * self.lam)

        # rewards-to-go computation
        self.ret_buf[path_slice] = self.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        # returns all data from buffer with normalized advantages

        assert self.ptr == self.max_size    # is buffer full ?
        self.ptr, self.path_start_idx = 0, 0

        adv_mean, adv_std, _, _ = get_stats(self.adv_buf)
        self.adv_buf = self.adv_buf - adv_mean
        if adv_std != 0:
            self.adv_buf /= adv_std
        return [self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf, self.logp_buf]

    def discount_cumsum(self, x, discount):
        # computes discounted cumulative sums of vectors.
        # input:  [x0, ..., xn-1, xn]
        # output: [..., xn-2 + d*xn-1 + d^2*xn, xn-1 + d*xn, xn]
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

