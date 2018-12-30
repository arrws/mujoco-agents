
import numpy as np
import tensorflow as tf
from gym.spaces import Box, Discrete
import scipy.signal


# NETWORK HELPERS
LOG_STD_MAX = 2
LOG_STD_MIN = -20
EPS = 1e-8

def gaussian_policy(x, a, hidden_sizes, activation, output_activation):
    act_dim = a.shape.as_list()[-1]
    net = mlp(x, list(hidden_sizes), activation, activation)
    mu = tf.layers.dense(net, act_dim, activation=output_activation)

    log_std = tf.layers.dense(net, act_dim, activation=tf.tanh)
    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    return mu, pi, logp_pi

def apply_squashing_func(mu, pi, logp_pi):
    mu = tf.tanh(mu)
    pi = tf.tanh(pi)
    # to avoid error strictly clip 1-pi**2 to [0,1] range
    logp_pi -= tf.reduce_sum(tf.log(clip_but_pass_gradient(1 - pi**2, l=0, u=1) + 1e-6), axis=1)
    return mu, pi, logp_pi


def actor_critic(x, a, hidden_sizes=(400,300), activation=tf.nn.relu,
                     output_activation=None, policy=gaussian_policy, act_space=None):
    # policy
    with tf.variable_scope('pi'):
        mu, pi, logp_pi = policy(x, a, hidden_sizes, activation, output_activation)
        mu, pi, logp_pi = apply_squashing_func(mu, pi, logp_pi)

    # make sure actions are in correct range
    action_scale = act_space.high[0]
    mu *= action_scale
    pi *= action_scale

    # vfs
    vf_mlp = lambda x : tf.squeeze(mlp(x, list(hidden_sizes)+[1], activation, None), axis=1)
    with tf.variable_scope('q1'):
        q1 = vf_mlp(tf.concat([x,a], axis=-1))
    with tf.variable_scope('q1', reuse=True):
        q1_pi = vf_mlp(tf.concat([x,pi], axis=-1))
    with tf.variable_scope('q2'):
        q2 = vf_mlp(tf.concat([x,a], axis=-1))
    with tf.variable_scope('q2', reuse=True):
        q2_pi = vf_mlp(tf.concat([x,pi], axis=-1))
    with tf.variable_scope('v'):
        v = vf_mlp(x)
    return mu, pi, logp_pi, q1, q2, q1_pi, q2_pi, v


def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def clip_but_pass_gradient(x, l=-1., u=1.):
    clip_up = tf.cast(x > u, tf.float32)
    clip_low = tf.cast(x < l, tf.float32)
    return x + tf.stop_gradient((u - x)*clip_up + (l - x)*clip_low)



# Simple Buffer Class for storing trajectories experience

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

