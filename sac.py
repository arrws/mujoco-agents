import numpy as np
import tensorflow as tf
import gym
import time

from common import *


"""
Soft Actor-Critic (like TD3)

"""


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




LOG_STD_MAX = 2
LOG_STD_MIN = -20

EPS = 1e-8

def gaussian_policy(x, a, hidden_sizes, activation, output_activation):
    act_dim = a.shape.as_list()[-1]
    net = mlp(x, list(hidden_sizes), activation, activation)
    mu = tf.layers.dense(net, act_dim, activation=output_activation)

    """
    Because algorithm maximizes trade-off of reward and entropy,
    entropy must be unique to state---and therefore log_stds need
    to be a neural network output instead of a shared-across-states
    learnable parameter vector. But for deep Relu and other nets,
    simply sticking an activationless dense layer at the end would
    be quite bad---at the beginning of training, a randomly initialized
    net could produce extremely large values for the log_stds, which
    would result in some actions being either entirely deterministic
    or too random to come back to earth. Either of these introduces
    numerical instability which could break the algorithm. To
    protect against that, we'll constrain the output range of the
    log_stds, to lie within [LOG_STD_MIN, LOG_STD_MAX]. This is
    slightly different from the trick used by the original authors of
    SAC---they used tf.clip_by_value instead of squashing and rescaling.
    I prefer this approach because it allows gradient propagation
    through log_std where clipping wouldn't, but I don't know if
    it makes much of a difference.
    """
    log_std = tf.layers.dense(net, act_dim, activation=tf.tanh)
    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    return mu, pi, logp_pi

def apply_squashing_func(mu, pi, logp_pi):
    mu = tf.tanh(mu)
    pi = tf.tanh(pi)
    # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
    logp_pi -= tf.reduce_sum(tf.log(clip_but_pass_gradient(1 - pi**2, l=0, u=1) + 1e-6), axis=1)
    return mu, pi, logp_pi


"""
Actor-Critics
"""
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


class Network:
    def __init__(self, hidden_sizes=(63,64), activation=tf.tanh, output_activation=None, policy=None, act_space=None, obs_space=None, gamma=0.99, polyak=0.995,  lr=1e-3, alpha=0.2):

        # Inputs to computation graph
        self.x_ph, self.a_ph, self.x2_ph, self.r_ph, self.d_ph = placeholders(obs_space.shape, act_space.shape, obs_space.shape, None, None)

        # Main outputs from computation graph
        with tf.variable_scope('main'):
            self.mu, self.pi, self.logp_pi, self.q1, self.q2, self.q1_pi, self.q2_pi, self.v = actor_critic(self.x_ph, self.a_ph, hidden_sizes, act_space = act_space)

        # Target value network
        with tf.variable_scope('target'):
            _, _, _, _, _, _, _, self.v_targ  = actor_critic(self.x2_ph, self.a_ph, hidden_sizes, act_space = act_space)

        # Min Double-Q:
        self.min_q_pi = tf.minimum(self.q1_pi, self.q2_pi)

        # Targets for Q and V regression
        self.q_backup = tf.stop_gradient(self.r_ph + gamma*(1-self.d_ph)*self.v_targ)
        self.v_backup = tf.stop_gradient(self.min_q_pi - alpha * self.logp_pi)

        # Soft actor-critic losses
        self.pi_loss = tf.reduce_mean(alpha * self.logp_pi - self.q1_pi)
        self.q1_loss = 0.5 * tf.reduce_mean((self.q_backup - self.q1)**2)
        self.q2_loss = 0.5 * tf.reduce_mean((self.q_backup - self.q2)**2)
        self.v_loss = 0.5 * tf.reduce_mean((self.v_backup - self.v)**2)
        self.value_loss = self.q1_loss + self.q2_loss + self.v_loss

        def get_vars(scope=''):
            return [x for x in tf.trainable_variables() if scope in x.name]

        # Policy train op
        # (has to be separate from value train op, because q1_pi appears in pi_loss)
        self.pi_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.train_pi_op = self.pi_optimizer.minimize(self.pi_loss, var_list=get_vars('main/pi'))

        # Value train op
        # (control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order)
        self.value_optimizer = tf.train.AdamOptimizer(learning_rate=lr)


        value_params = get_vars('main/q') + get_vars('main/v')
        with tf.control_dependencies([self.train_pi_op]):
            self.train_value_op = self.value_optimizer.minimize(self.value_loss, var_list=value_params)

        # Polyak averaging for target variables
        # (control flow because sess.run otherwise evaluates in nondeterministic order)
        with tf.control_dependencies([self.train_value_op]):
            self.target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                                      for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

        # All ops to call during one training step
        self.step_ops = [self.pi_loss, self.q1_loss, self.q2_loss, self.v_loss, self.q1, self.q2, self.v, self.logp_pi,
                    self.train_pi_op, self.train_value_op, self.target_update]

        # Initializing targets to match main variables
        self.target_init = tf.group([tf.assign(v_targ, v_main)
                                  for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])


def sac(env_name, kwargs=dict(), steps_per_epoch=5000, epochs=100, replay_size=int(1e6), gamma=0.99, polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000, max_ep_len=1000):

    logger = Logger()

    # Environment
    env, test_env = gym.make(env_name), gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    kwargs['obs_space'] = env.observation_space
    kwargs['act_space'] = env.action_space
    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Experience Replay
    buf = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Network
    print(kwargs)
    net = Network(**kwargs)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(net.target_init)


    def get_action(o, deterministic=False):
        act_op = net.mu if deterministic else net.pi
        return sess.run(act_op, feed_dict={net.x_ph: o.reshape(1,-1)})[0]

    def test_agent(n=10):
        global sess, pi, q1, q2, q1_pi, q2_pi
        for j in range(n):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time
                o, r, d, _ = test_env.step(get_action(o, True))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    total_steps = steps_per_epoch * epochs

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards,
        use the learned policy.
        """
        if t > start_steps:
            a = get_action(o)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        buf.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        if d or (ep_len == max_ep_len):
            """
            Perform all SAC updates at the end of the trajectory.
            This is a slight difference from the SAC specified in the
            original paper.
            """
            for j in range(ep_len):
                batch = buf.sample_batch(batch_size)
                feed_dict = {net.x_ph: batch['obs1'],
                             net.x2_ph: batch['obs2'],
                             net.a_ph: batch['acts'],
                             net.r_ph: batch['rews'],
                             net.d_ph: batch['done'],
                            }
                outs = sess.run(net.step_ops, feed_dict)
                logger.store(LossPi=outs[0], LossQ1=outs[1], LossQ2=outs[2],
                             LossV=outs[3], Q1Vals=outs[4], Q2Vals=outs[5],
                             VVals=outs[6], LogPi=outs[7])

            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0


        # End of epoch wrap-up
        if t > 0 and t % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            # # Save model
            # if (epoch % save_freq == 0) or (epoch == epochs-1):
            #     logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log('Epoch', epoch)
            logger.log('EpRet', with_min_and_max=True)
            logger.log('TestEpRet', with_min_and_max=True)
            logger.log('EpLen', average_only=True)
            logger.log('TestEpLen', average_only=True)
            logger.log('TotalEnvInteracts', t)
            logger.log('Q1Vals', with_min_and_max=True)
            logger.log('Q2Vals', with_min_and_max=True)
            logger.log('VVals', with_min_and_max=True)
            logger.log('LogPi', with_min_and_max=True)
            logger.log('LossPi', average_only=True)
            logger.log('LossQ1', average_only=True)
            logger.log('LossQ2', average_only=True)
            logger.log('LossV', average_only=True)
            logger.log('Time', time.time()-start_time)
            print("")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='MountainCarContinuous-v0')
    parser.add_argument('--hid', type=int, default=300)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()

    sac(env_name = args.env, kwargs=dict(hidden_sizes=[args.hid]*args.layers),
        gamma=args.gamma, epochs=args.epochs)

