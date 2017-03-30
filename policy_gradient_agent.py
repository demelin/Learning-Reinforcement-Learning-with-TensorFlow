""" All code in this project follows the tutorials provided in the Simple RL series, part of which is found here:
https://medium.com/@awjuliani/super-simple-reinforcement-learning-tutorial-part-2-ded33892c724#.f0442wu0m """

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xi

# Import the OpenAi Gym environment
import gym

env = gym.make('CartPole-v0')

"""
# Trying out random episodes
env.reset()
random_episodes = 0
reward_sum = 0

# Execute episodes of random bahaviour
while random_episodes < 10:
    env.render()
    observation, reward, done, _ = env.step(np.random.randint(0, 2))
    reward_sum += reward
    # Check if episode has terminated
    if done:
        random_episodes += 1
        print('Reward for this episode was %.4f' % reward_sum)
        # Reset environment and tracking variables
        reward_sum = 0
        env.reset()
"""

# Declare hyperparameters for the agent network and helper functions
HIDDEN_SIZE = 8
LR = 1e-2
GAMMA = 0.99
INPUT_DIMS = 4
NUM_ACTIONS = 2
# Define training parameters
total_episodes = 5000
max_steps = 999
update_frequency = 5


# Define helper function to calculate discounted rewards
def discount_rewards(r):
    """ Calculates discounted on-policy rewards. """
    discounted_r = np.zeros_like(r)
    running_add = 0
    # Weights closer to end seen as negative, the ones further away from task failure as positive
    for t in reversed(range(0, r.size)):
        running_add = running_add * GAMMA + r[t]
        discounted_r[t] = running_add
    return discounted_r


# Define agent class
class PolicyGradAgent(object):
    """ Simple RL agent trained on policy. """

    def __init__(self, input_dims, hidden_size, num_actions, learning_rate):
        # Initialize args
        self.input_dims = input_dims
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.learning_rate = learning_rate

        # Placeholder for observations, actions, rewards
        self.state = tf.placeholder(shape=[None, self.input_dims], dtype=tf.float32, name='state')
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32, name='actions')
        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32, name='rewards')

        # Define layers etc
        with tf.variable_scope('layer_1'):
            w_1 = tf.get_variable(shape=[self.input_dims, self.hidden_size], dtype=tf.float32, initializer=xi(),
                                  name='weight_1')
            o_1 = tf.nn.relu(tf.matmul(self.state, w_1), name='out_1')
        with tf.variable_scope('layer_2'):
            w_2 = tf.get_variable(shape=[self.hidden_size, self.num_actions], dtype=tf.float32, initializer=xi(),
                                  name='weight_2')
            self.probabilities = tf.nn.softmax(tf.matmul(o_1, w_2), name='probabilities')

        # Loss computation
        with tf.variable_scope('loss'):
            indices = tf.range(0, tf.shape(self.probabilities)[0]) * tf.shape(self.probabilities)[1] + \
                      self.action_holder
            responsible_outputs = tf.gather(tf.reshape(self.probabilities, [-1]), indices)
            self.loss = - tf.reduce_mean(tf.multiply(tf.log(responsible_outputs), self.reward_holder))

            self.t_vars = tf.trainable_variables()
            # Instantiate gradient holders
            self.gradient_holders = list()
            for _idx, var in enumerate(self.t_vars):
                placeholder = tf.placeholder(dtype=tf.float32, name=str(_idx) + '_holder')
                self.gradient_holders.append(placeholder)

            # Obtain gradients
            self.gradients = tf.gradients(self.loss, self.t_vars)

            # Optimize
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, self.t_vars))


# Initialize the agent
agent = PolicyGradAgent(INPUT_DIMS, HIDDEN_SIZE, NUM_ACTIONS, LR)

# Run the session
with tf.Session() as sess:
    # Initialize graph variables
    sess.run(tf.global_variables_initializer())
    # Tracking variables
    e = 0
    total_reward = list()
    # Initialize gradient buffer
    grad_buffer = sess.run(tf.trainable_variables())
    for idx, grad in enumerate(grad_buffer):
        grad_buffer[idx] = grad * 0

    # Initiate training loop
    while e < total_episodes:
        if total_reward[-1] > 180:
            env.render()

        curr_state = env.reset()
        solved = False
        running_reward = 0
        episode_history = list()
        for j in range(max_steps):
            # Choose action
            curr_state = np.reshape(curr_state, [1, INPUT_DIMS])
            action_distribution = sess.run(agent.probabilities, feed_dict={agent.state: curr_state})
            # Add some randomness to the choice for exploration
            action_value = np.random.choice(action_distribution[0], p=action_distribution[0])
            match = np.square(action_distribution - action_value)
            action = np.argmin(match)

            # Perform step and memorize transition
            next_state, reward, done, info = env.step(action)
            episode_history.append([curr_state, action, reward, next_state])
            curr_state = next_state
            running_reward += reward

            # Check if episode has completed, then update network
            if done:
                episode_history = np.array(episode_history)
                # Discount rewards
                episode_history[:, 2] = discount_rewards(episode_history[:, 2])
                feed_dict = {agent.reward_holder: episode_history[:, 2],
                             agent.action_holder: episode_history[:, 1],
                             agent.state: np.vstack(episode_history[:, 0])}
                # Get gradient for current episode
                gradients = sess.run(agent.gradients, feed_dict=feed_dict)
                # Append to grad buffer
                for idx, grad in enumerate(gradients):
                    grad_buffer[idx] = grad_buffer[idx] + grad

                if e % update_frequency == 0 and e != 0:
                    feed_dict = dict(zip(agent.gradient_holders, grad_buffer))
                    _ = sess.run(agent.update_batch, feed_dict=feed_dict)
                    for idx, grad in enumerate(grad_buffer):
                        grad_buffer[idx] = grad * 0

                # Report some results
                if e % (update_frequency * 10) and e != 0:
                    avg_r = np.mean(total_reward[-update_frequency * 10:])
                    print('Average episode reward since last check: %.4f' % avg_r)
                    if avg_r >= 200:
                        print('Solution found in %d steps' % len(total_reward))
                        solved = True

                total_reward.append(running_reward)
                # Terminate current episode loop following completion or after max_steps
                break

        e += 1
        if solved:
            print('Total episodes played: %d' % e)
            break
