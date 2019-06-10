#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Implementation of DDPG - Deep Deterministic Policy Gradient
Algorithm and hyperparameter details can be found here:
    http://arxiv.org/pdf/1509.02971v2.pdf
The algorithm is tested on the Pendulum-v0 OpenAI gym task
and developed with tflearn + Tensorflow
Author: Patrick Emami
"""
import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
import gym_sat_inspection
import tflearn
import argparse
import pprint as pp
import util
import math
from replay_buffer import ReplayBuffer


# In[2]:
class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.
    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size
        self.action_bound = action_bound
        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[
                                     len(self.network_params):]
        self.saver = tf.train.Saver()

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate). \
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.fully_connected(inputs, 400)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 300)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(
            net, self.a_dim, activation='tanh', weights_init=w_init)
        scaled_out = tf.multiply(out, self.action_bound)
        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.
    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
                                                  + tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        net = tflearn.fully_connected(inputs, 400)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, 300)
        t2 = tflearn.fully_connected(action, 300)

        net = tflearn.activation(
            tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)






# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.5, theta=.15, dt=1, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

# ===========================
#   Tensorflow Summary Ops
# ===========================

def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars


# In[6]:



# ===========================
#   Agent Training
# ===========================

def train(sess, env, args, actor, critic, actor_noise):
    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()
    start_steps = int(float(args['max_episodes'])/5)
    zero_steps = -1
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(args['summary_dir'], sess.graph)

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))

    # Needed to enable BatchNorm.
    # This hurts the performance on Pendulum but could be useful
    # in other environments.
    # tflearn.is_training(True)

    for i in range(int(args['max_episodes'])):

        s = env.reset()

        ep_reward = 0
        ep_ave_max_q = 0

        for j in range(int(args['max_episode_len'])):

            if args['render_env']:
                env.render()

            # Added exploration noise
            # a = actor.predict(np.reshape(s, (1, 3))) + (1. / (1. + i))
            if i > start_steps:
                a = actor.predict(np.reshape(s, (1, actor.s_dim))) + actor_noise()
                s2, r, terminal, info = env.step(a[0])
            elif i >zero_steps:
                mean = np.array([0,0,0,0])
                cov = np.array([[0.25, 0,0,0], [0, 0.25,0,0],[0,0,0.25,0],[0, 0, 0, 0.25]])
                a = np.random.multivariate_normal(mean, cov, 1)
                a = a[0]
                for k in range(len(a)):
                    val = a[k]
                    if val >=0.5:
                        a[k] = 0.5
                    elif val <=-0.5:
                        a[k] = -0.5
                s2, r, terminal, info = env.step(a)
            else:
                a = np.array([0,0,0,0])
                s2, r, terminal, info = env.step(a)

            replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r,
                              terminal, np.reshape(s2, (actor.s_dim,)))

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > int(args['minibatch_size']):
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(int(args['minibatch_size']))

                # Calculate targets
                target_q = critic.predict_target(
                    s2_batch, actor.predict_target(s2_batch))

                y_i = []
                for k in range(int(args['minibatch_size'])):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + critic.gamma * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(
                    s_batch, a_batch, np.reshape(y_i, (int(args['minibatch_size']), 1)))

                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            s = s2
            ep_reward += r

            if terminal:
                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: ep_reward,
                    summary_vars[1]: ep_ave_max_q / float(j+1)
                })

                writer.add_summary(summary_str, i)
                writer.flush()
                break
        print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f}'.format(int(ep_reward), i, (ep_ave_max_q / float(j+1))))



def test(sess, env, args, actor):
    # Set up summary Ops
    reward_list = []
    action_list = []
    pct_list = []
    for i in range(int(args['test_episodes'])):

        s = env.reset()
        ep_reward = 0

        for j in range(int(args['max_episode_len'])):
            # a = actor.predict(np.reshape(s, (1, 3))) + (1. / (1. + i))
            a = actor.predict(np.reshape(s, (1, actor.s_dim)))
            s2, r, terminal, info = env.step(a[0])
            s = s2
            ep_reward += r
            action_list.append(a)
            if terminal:
                print('| Reward: {:d} | Episode: {:d}'.format(int(ep_reward),i))
                reward_list.append(int(ep_reward))
                pct_list.append(env.percent_coverage)
                break
    return reward_list,action_list,pct_list

def test_rand_policy(env, args):
    # Set up summary Ops
    reward_list = []
    action_list = []
    pct_list = []
    for i in range(int(args['test_episodes'])):

        s = env.reset()
        ep_reward = 0

        for j in range(int(args['max_episode_len'])):
            # a = actor.predict(np.reshape(s, (1, 3))) + (1. / (1. + i))
            mean = np.array([0,0,0,0])
            cov = np.array([[0.25, 0,0,0], [0, 0.25,0,0],[0,0,0.25,0],[0, 0, 0, 0.25]])
            a = np.random.multivariate_normal(mean, cov, 1)
            a = a[0]
            for k in range(len(a)):
                val = a[k]
                if val >=0.5:
                    a[k] = 0.5
                elif val <=-0.5:
                    a[k] = -0.5
            s2, r, terminal, info = env.step(a)
            s = s2
            ep_reward += r
            action_list.append(a)
            if terminal:
                print('| Reward: {:d} | Episode: {:d}'.format(int(ep_reward),i))
                reward_list.append(int(ep_reward))
                pct_list.append(env.percent_coverage)
                break
    return reward_list,action_list,pct_list

def test_zero_policy(env, args):
    # Set up summary Ops
    reward_list = []
    action_list = []
    pct_list = []
    for i in range(int(args['test_episodes'])):

        s = env.reset()
        ep_reward = 0

        for j in range(int(args['max_episode_len'])):
            # a = actor.predict(np.reshape(s, (1, 3))) + (1. / (1. + i))
            a = np.array([0,0,0,0])
            s2, r, terminal, info = env.step(a)
            s = s2
            ep_reward += r
            action_list.append(a)
            if terminal:
                print('| Reward: {:d} | Episode: {:d}'.format(int(ep_reward),i))
                reward_list.append(int(ep_reward))
                pct_list.append(env.percent_coverage)
                break
    return reward_list,action_list,pct_list


def main(args):
    oe_insp_0 = np.array([6771., 0.0, 0.0, 0.000001, math.radians(5.0), 0.0])
    oe_targ_0 = np.array([6771., 0.0, 0.0, 0.0, math.radians(5.0), 0.0])
    roe_0 = util.ROEfromOEs(oe_insp_0, oe_targ_0)
    dt = 500
    RTN_0 = util.ROE2HILL(roe_0, oe_targ_0[0], oe_targ_0[1])
    w = math.sqrt(398600 / math.pow(oe_targ_0[0], 3))
    print("W = ", w)
    w_0 = np.array([0, 0, -w])
    Pg_0 = np.array([np.linalg.norm(RTN_0), 0, 0])
    # Array to store "feature points" by theta, phi pair in Geometric frame.
    map_0 = np.zeros((360, 360))
    num_photos = args['max_episode_len']
    tau = 50
    fuel_0 = np.array((0.1,))
    state_0 = np.concatenate((roe_0, Pg_0, w_0, fuel_0), axis=0)
    kwargs = {'init_state': state_0, 'map': map_0, 'inspOE_0': oe_insp_0, 'targOE_0': oe_targ_0, 'RTN_0': RTN_0,
              'dt': dt, 'num_photos': num_photos, 'tau': tau}
    with tf.Session() as sess:

        env = gym.make(args['env'], **kwargs)
        np.random.seed(int(args['random_seed']))
        tf.set_random_seed(int(args['random_seed']))
        env.seed(int(args['random_seed']))

        state_dim = len(state_0)
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high[0]
        assert (env.action_space.high[0] == -env.action_space.low[0])
        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                             float(args['actor_lr']), float(args['tau']),
                             int(args['minibatch_size']))

        critic = CriticNetwork(sess, state_dim, action_dim,
                               float(args['critic_lr']), float(args['tau']),
                               float(args['gamma']),
                               actor.get_num_trainable_vars())

        actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

        if args['use_gym_monitor']:
            if not args['render_env']:
                env = wrappers.Monitor(
                    env, args['monitor_dir'], video_callable=False, force=True)
            else:
                env = wrappers.Monitor(env, args['monitor_dir'], force=True)
        print('Beginning Training')
        train(sess, env, args, actor, critic, actor_noise)
        save_path = actor.saver.save(sess, "./results/model.ckpt")
        if args['use_gym_monitor']:
            env.monitor.close()
        print('Beginning Testing')
        """
            new_saver = tf.train.import_meta_graph('./results/model.ckpt.meta')
            new_saver.restore(sess, tf.train.latest_checkpoint('./results/'))
        """
        reward_list, action_list, pct_list = test(sess, env, args, actor)
        print('Reward List', reward_list)
        print('Actions List', action_list)
        print('Average Reward', np.average(reward_list))
        print('Average Pct Coverage', np.average(pct_list))

        print('Testing a Random Policy')
        rand_list, rand_actions, pct_list_rand = test_rand_policy(env, args)
        print('Reward List for Random Policy', rand_list)
        print('Actions List for Random Policy ', rand_actions)
        print('Average Reward for Random Policy ', np.average(rand_list))
        print('Average Pct Coverage for Random Policy ', np.average(pct_list_rand))

        print('Testing a Zero Policy')
        zero_list, zero_actions, pct_list_zero = test_zero_policy(env, args)
        print('Reward List for Zero Policy', zero_list)
        print('Actions List for Zero Policy ', zero_actions)
        print('Average Reward for Zero Policy ', np.average(zero_list))
        print('Average Pct Coverage for Zero Policy ', np.average(pct_list_zero))


actor_lr = 0.0001
critic_lr = 0.001
gamma = 0.99
tau = 0.001
buffer_size = 100000
minibatch_size = 64
env = 'sat_inspection-v0'
random_seed = 1234
max_episodes = 3000
max_episode_len = 100
render_env = False
use_gym_monitor = False
test_episodes = 100
monitor_dir = './results/gym_ddpg'
summary_dir = './results/tf_ddpg'
args = {'actor_lr': actor_lr, 'critic_lr': critic_lr,'gamma': gamma,'tau': tau,'buffer_size': buffer_size,'minibatch_size': minibatch_size,
       'env': env, 'random_seed': random_seed, 'max_episodes': max_episodes,'max_episode_len': max_episode_len,'render_env': render_env,
       'use_gym_monitor': use_gym_monitor, 'monitor_dir': monitor_dir, 'summary_dir': summary_dir, 'test_episodes': test_episodes}
main(args)