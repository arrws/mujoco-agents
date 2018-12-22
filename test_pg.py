import tensorflow as tf
import numpy as np
import gym
import argparse


class Network:
    def __init__(self, sizes, obs_dim, n_acts, lr, activation=tf.tanh, output_activation=None):

        # policy network
        self.obs_ph = tf.placeholder(shape=(None, obs_dim), dtype=tf.float32)
        x = self.obs_ph
        for size in sizes[:-1]:
            x = tf.layers.dense(x, units=size, activation=activation)
        logits = tf.layers.dense(x, units=sizes[-1], activation=output_activation)
        # sample actions
        self.actions = tf.squeeze(tf.multinomial(logits=logits, num_samples=1), axis=1)

        # loss function
        self.weights_ph = tf.placeholder(shape=(None,), dtype=tf.float32)
        self.act_ph = tf.placeholder(shape=(None,), dtype=tf.int32)
        action_masks = tf.one_hot(self.act_ph, n_acts)
        log_probs = tf.reduce_sum(action_masks * tf.nn.log_softmax(logits), axis=1)
        self.loss = -tf.reduce_mean(self.weights_ph * log_probs)

        # train op
        self.train = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)


def reward_to_go(rewards):
    n = len(rewards)
    rtgs = np.zeros_like(rewards)
    for i in reversed(range(n)):
        rtgs[i] = rewards[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs


def train(env_name='CartPole-v0', hidden_sizes=[32], lr=1e-2, epochs=5, batch_size=5000, render=True):

    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    net = Network(sizes=hidden_sizes+[n_acts], obs_dim=obs_dim, n_acts=n_acts, lr=lr)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())


    def train_one_epoch():
        batch_obs = []          # observations
        batch_acts = []         # actions
        batch_weights = []      # R(tau) weighting
        batch_rets = []         # episode returns
        batch_lens = []         # episode lengths
        rewards = []            # rewards per step

        obs = env.reset()
        done = False

        first_ep = True

        while True:
            if first_ep and render:
                env.render()

            batch_obs.append(obs.copy())

            a = sess.run(net.actions, { net.obs_ph: obs.reshape(1,-1) })[0]
            obs, r, done, _ = env.step(a)

            batch_acts.append(a)
            rewards.append(r)

            if done:
                ep_ret, ep_len = sum(rewards), len(rewards)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                batch_weights += [ep_ret] * ep_len              # full R(tau)
                # batch_weights += list(reward_to_go(rewards))    # R to go

                # reset
                obs, done, rewards = env.reset(), False, []
                first_ep = False

                if len(batch_obs) > batch_size:
                    # if not enough experience
                    break

        # policy gradient update step
        batch_loss, _ = sess.run([net.loss, net.train],
                                 feed_dict={
                                    net.obs_ph: np.array(batch_obs),
                                    net.act_ph: np.array(batch_acts),
                                    net.weights_ph: np.array(batch_weights)
                                 })
        return batch_loss, batch_rets, batch_lens

    # main loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))


if __name__ == '__main__':
    print('\nVanilla Policy Gradient.\n')
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', type=bool, default=True)
    parser.add_argument('--epochs', type=str, default=5)
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    train(env_name=args.env_name, render=args.render, lr=args.lr)

