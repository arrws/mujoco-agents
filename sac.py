import numpy as np
import tensorflow as tf
import gym
import time

from common import *
from sac_utils import *


"""
Soft Actor-Critic (like TD3)

"""

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


    def get_action(obs, deterministic=False):
        act_op = net.mu if deterministic else net.pi
        return sess.run(act_op, feed_dict={net.x_ph: obs.reshape(1,-1)})[0]

    def test_agent(n=10):
        global sess, pi, q1, q2, q1_pi, q2_pi
        for j in range(n):
            obs, r, done, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not(done or (ep_len == max_ep_len)):
                # Take deterministic actions at test time
                obs, r, done, _ = test_env.step(get_action(obs, True))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)



    start_time = time.time()
    obs, r, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    total_steps = steps_per_epoch * epochs

    # MAIN LOOP
    for step in range(total_steps):
        # random sample while start_steps then learned policy
        if step > start_steps:
            a = get_action(obs)
        else:
            a = env.action_space.sample()

        obs2, r, done, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # ignore the "done" signal if not based on the agent's state
        done = False if ep_len==max_ep_len else done

        buf.store(obs, a, r, obs2, done)
        obs = obs2

        if done or (ep_len == max_ep_len):
            # Perform all SAC updates at the end of the trajectory.
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
            obs, r, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0


        # End of epoch wrap-up
        if step > 0 and step % steps_per_epoch == 0:
            epoch = step // steps_per_epoch

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log('Epoch', epoch)
            logger.log('EpRet', with_min_and_max=True)
            logger.log('TestEpRet', with_min_and_max=True)
            logger.log('EpLen', average_only=True)
            logger.log('TestEpLen', average_only=True)
            logger.log('TotalEnvInteracts', step)
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

