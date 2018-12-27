import numpy as np
import tensorflow as tf
import gym
from gym.spaces import Box, Discrete
import time

from common import *


"""
Proximal Policy Optimization CLIP (with aprox KL early stopping)

"""


class Network:
    def __init__(self, hidden_sizes=(63,64), activation=tf.tanh, output_activation=None, policy=None, act_space=None, obs_space=None, gamma=0.99, clip_ratio=0.2,  pi_lr=3e-4, vf_lr=1e-3):
        print(act_space, obs_space)

        # Inputs to computation graph
        self.x_ph, self.a_ph = placeholders_from_spaces(obs_space, act_space)
        self.adv_ph, self.ret_ph, self.logp_ph = placeholders(None, None, None)
        # easy retrieve [obs, actions, advantage, returns, logprob]
        self.all_phs = [self.x_ph, self.a_ph, self.adv_ph, self.ret_ph, self.logp_ph]

        policy = get_policy(act_space)
        self.pi, self.logp, self.logp_pi = policy(self.x_ph, self.a_ph, hidden_sizes, activation, output_activation, act_space)
        self.v = tf.squeeze(mlp(self.x_ph, list(hidden_sizes)+[1], activation, None), axis=1)

        # easy retrieve [action, value, and logprob]
        self.get_action_ops = [self.pi, self.v, self.logp_pi]

        # PPO objectives
        ratio = tf.exp(self.logp - self.logp_ph)          # pi(a|s) / pi_old(a|s)
        min_adv = tf.where(self.adv_ph>0, (1+clip_ratio)*self.adv_ph, (1-clip_ratio)*self.adv_ph)
        self.pi_loss = -tf.reduce_mean(tf.minimum(ratio * self.adv_ph, min_adv))
        self.v_loss = tf.reduce_mean((self.ret_ph - self.v)**2)

        # info for trainning
        self.approx_kl = tf.reduce_mean(self.logp_ph - self.logp)     # sample estimate for kl-divergence
        self.approx_ent = tf.reduce_mean(-self.logp)                  # sample estimate for entropy
        self.clipped = tf.logical_or(ratio > (1+clip_ratio), ratio < (1-clip_ratio))
        self.clipfrac = tf.reduce_mean(tf.cast(self.clipped, tf.float32))

        # optimizers
        self.train_pi = tf.train.AdamOptimizer(learning_rate=pi_lr).minimize(self.pi_loss)
        self.train_v = tf.train.AdamOptimizer(learning_rate=vf_lr).minimize(self.v_loss)



def ppo(env_name, kwargs=dict(), steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4, vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000, target_kl=0.01):

    logger = Logger()

    # Environment
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    kwargs['obs_space'] = env.observation_space
    kwargs['act_space'] = env.action_space

    # Experience Replay
    buf = Buffer(obs_dim, act_dim, steps_per_epoch, gamma, lam)

    # Network
    print(kwargs)
    net = Network(**kwargs)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())


    def update():
        inputs = {k:v for k,v in zip(net.all_phs, buf.get())}
        pi_loss, v_loss, ent = sess.run([net.pi_loss, net.v_loss, net.approx_ent], feed_dict=inputs)

        # Policy gradient step
        for i in range(train_pi_iters):
            _, kl = sess.run([net.train_pi, net.approx_kl], feed_dict=inputs)
            if kl > 1.5 * target_kl:
                print('Early stopping at step %d due to reaching max kl.'%i)
                break
        logger.store(StopIter=i)

        # Value function optimization step
        for _ in range(train_v_iters):
            sess.run(net.train_v, feed_dict=inputs)

        # log changes from update
        pi_loss_new, v_loss_new, kl, cf = sess.run([net.pi_loss, net.v_loss, net.approx_kl, net.clipfrac], feed_dict=inputs)
        logger.store(LossPi=pi_loss,
                     LossV=v_loss,
                     KL=kl, Entropy=ent,
                     ClipFrac=cf,
                     dLossPi=(pi_loss_new - pi_loss),
                     dLossV=(v_loss_new - v_loss))


    start_time = time.time()
    obs, r, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0

    # MAIN LOOP
    for epoch in range(epochs):
        for step in range(steps_per_epoch):

            # get next action
            a, v_t, logp_t = sess.run(net.get_action_ops, feed_dict={net.x_ph: obs.reshape(1,-1)})

            buf.store(obs, a, r, v_t, logp_t)
            logger.store(VVals=v_t)

            obs, r, done, _ = env.step(a[0])
            ep_ret += r
            ep_len += 1

            if done or (ep_len == max_ep_len) or (step == steps_per_epoch-1):
                # if trajectory didn't reach terminal state, bootstrap value target
                last_val = r if done else sess.run(net.v, feed_dict={net.x_ph: obs.reshape(1,-1)})
                buf.finish_path(last_val)

                if done or (ep_len == max_ep_len):
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                else:
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len)

                obs, r, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0


        # perform PPO update
        update()

        # Log info about epoch
        logger.log('Epoch', epoch)
        logger.log('EpRet', with_min_and_max=True)
        logger.log('EpLen', average_only=True)
        logger.log('VVals', with_min_and_max=True)
        logger.log('EnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log('LossPi', average_only=True)
        logger.log('LossV', average_only=True)
        logger.log('dLossPi', average_only=True)
        logger.log('dLossV', average_only=True)
        logger.log('Entropy', average_only=True)
        logger.log('KL', average_only=True)
        logger.log('ClipFrac', average_only=True)
        logger.log('StopIter', average_only=True)
        logger.log('Time', time.time()-start_time)
        print("")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()

    ppo(env_name = args.env, kwargs=dict(hidden_sizes=[args.hid]*args.layers),
        gamma=args.gamma, steps_per_epoch=args.steps, epochs=args.epochs)
