import numpy as np
import tensorflow as tf
import gym
from gym.spaces import Box, Discrete
import time

from common import *

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



class Network:
    def __init__(self, hidden_sizes=(63,64), activation=tf.tanh, output_activation=None, policy=None, action_space=None, observation_space=None, pi_lr=3e-4, vf_lr=1e-3, ):
        print(action_space, observation_space)

        # Inputs to computation graph
        self.x_ph, self.a_ph = placeholders_from_spaces(observation_space, action_space)
        self.adv_ph, self.ret_ph, self.logp_old_ph = placeholders(None, None, None)


        if policy is None and isinstance(action_space, Box):
            policy = self.mlp_gaussian_policy
        elif policy is None and isinstance(action_space, Discrete):
            policy = self.mlp_categorical_policy
        # with tf.variable_scope('pi'):
        # with tf.variable_scope('v'):
        self.pi, self.logp, self.logp_pi = policy(self.x_ph, self.a_ph, hidden_sizes, activation, output_activation, action_space)
        self.v = tf.squeeze(self.mlp(self.x_ph, list(hidden_sizes)+[1], activation, None), axis=1)


        # need all placeholders in *this* order later (to zip with data from buffer)
        self.all_phs = [self.x_ph, self.a_ph, self.adv_ph, self.ret_ph, self.logp_old_ph]

        # every step, get: action, value, and logprob
        self.get_action_ops = [self.pi, self.v, self.logp_pi]

        # vpg objectives
        self.pi_loss = -tf.reduce_mean(self.logp * self.adv_ph)
        self.v_loss = tf.reduce_mean((self.ret_ph - self.v)**2)

        # info (useful to watch during learning)
        self.approx_kl = tf.reduce_mean(self.logp_old_ph - self.logp)      # a sample estimate for kl-divergence, easy to compute
        self.approx_ent = tf.reduce_mean(-self.logp)                  # a sample estimate for entropy, also easy to compute

        # optimizers
        self.train_pi = tf.train.AdamOptimizer(learning_rate=pi_lr).minimize(self.pi_loss)
        self.train_v = tf.train.AdamOptimizer(learning_rate=vf_lr).minimize(self.v_loss)

    def mlp(self, x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
        for h in hidden_sizes[:-1]:
            x = tf.layers.dense(x, units=h, activation=activation)
        return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)


    def mlp_categorical_policy(self, x, a, hidden_sizes, activation, output_activation, action_space):
        act_dim = action_space.n
        logits = self.mlp(x, list(hidden_sizes)+[act_dim], activation, None)
        logp_all = tf.nn.log_softmax(logits)
        pi = tf.squeeze(tf.multinomial(logits,1), axis=1)
        logp = tf.reduce_sum(tf.one_hot(a, depth=act_dim) * logp_all, axis=1)
        logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1)
        return pi, logp, logp_pi


    def mlp_gaussian_policy(self, x, a, hidden_sizes, activation, output_activation, action_space):
        act_dim = a.shape.as_list()[-1]
        mu = mlp(x, list(hidden_sizes)+[act_dim], activation, output_activation)
        log_std = tf.get_variable(name='log_std', initializer=-0.5*np.ones(act_dim, dtype=np.float32))
        std = tf.exp(log_std)
        pi = mu + tf.random_normal(tf.shape(mu)) * std
        logp = gaussian_likelihood(a, mu, log_std)
        logp_pi = gaussian_likelihood(pi, mu, log_std)
        return pi, logp, logp_pi



"""
Vanilla Policy Gradient
(with GAE-Lambda for advantage estimation)
"""

def vpg(env_fn, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=50, gamma=0.99, train_v_iters=80, lam=0.97, max_ep_len=1000,
        logger_kwargs=dict(), save_freq=10):

    logger = Logger()
    # logger.save_config(locals())

    seed += 10000 * 132#proc_id()
    tf.set_random_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space
    ac_kwargs['observation_space'] = env.observation_space

    # Experience buffer
    local_steps_per_epoch = int(steps_per_epoch)# / num_procs())
    buf = Buffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # Count variables
    def get_vars(scope=''):
        return [x for x in tf.trainable_variables() if scope in x.name]
    def count_vars(scope=''):
        v = get_vars(scope)
        return sum([np.prod(var.shape.as_list()) for var in v])
    var_counts = tuple(count_vars(scope) for scope in ['pi', 'v'])
    print('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)


    print(ac_kwargs)
    net = Network(**ac_kwargs)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Setup model saving
    # logger.setup_tf_saver(sess, inputs={'x': net.x_ph}, outputs={'pi': net.pi, 'v': net.v})

    def update():
        inputs = {k:v for k,v in zip(net.all_phs, buf.get())}
        pi_l_old, v_l_old, ent = sess.run([net.pi_loss, net.v_loss, net.approx_ent], feed_dict=inputs)

        # Policy gradient step
        sess.run(net.train_pi, feed_dict=inputs)

        # Value function learning
        for _ in range(train_v_iters):
            sess.run(net.train_v, feed_dict=inputs)

        # Log changes from update
        pi_l_new, v_l_new, kl = sess.run([net.pi_loss, net.v_loss, net.approx_kl], feed_dict=inputs)
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent,
                     DeltaLossPi=(pi_l_new - pi_l_old),
                     DeltaLossV=(v_l_new - v_l_old))

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            a, v_t, logp_t = sess.run(net.get_action_ops, feed_dict={net.x_ph: o.reshape(1,-1)})

            # save and log
            buf.store(o, a, r, v_t, logp_t)
            logger.store(VVals=v_t)

            o, r, d, _ = env.step(a[0])
            ep_ret += r
            ep_len += 1

            terminal = d or (ep_len == max_ep_len)
            if terminal or (t==local_steps_per_epoch-1):
                if not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len)
                # if trajectory didn't reach terminal state, bootstrap value target
                last_val = r if d else sess.run(net.v, feed_dict={net.x_ph: o.reshape(1,-1)})
                buf.finish_path(last_val)

                if terminal:
                #     # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)

                o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        # # Save model
        # if (epoch % save_freq == 0) or (epoch == epochs-1):
        #     logger.save_state({'env': env}, None)

        # Perform VPG update!
        update()

        # Log info about epoch
        logger.log('Epoch', epoch)
        logger.log('EpRet', with_min_and_max=True)
        logger.log('EpLen', average_only=True)
        logger.log('VVals', with_min_and_max=True)
        logger.log('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log('LossPi', average_only=True)
        logger.log('LossV', average_only=True)
        logger.log('DeltaLossPi', average_only=True)
        logger.log('DeltaLossV', average_only=True)
        logger.log('Entropy', average_only=True)
        logger.log('KL', average_only=True)
        logger.log('Time', time.time()-start_time)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='vpg')
    args = parser.parse_args()


    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    vpg(lambda : gym.make(args.env),
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma,
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs)