""" A dual network policy gradient RL architecture: Model network learns a representation of the environment on the
basis of observations it receives from the interactions between the PolicyNet - encoding the agent - and the true
environment. PolicyNet learns its optimal policy by learning from the simulated data provided by the ModelNet only.
The so defined procedure accelerates the training process for the agent. """

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xi
import gym

env = gym.make('CartPole-v0')

# Declare hyperparameters for the agent network and helper functions
# Current parameters work sufficiently well depending on network initialization
AGENT_HIDDEN_1 = 64
AGENT_HIDDEN_2 = 128
AGENT_HIDDEN_3 = 32
MODEL_HIDDEN_1 = 256
MODEL_HIDDEN_2 = 512
MODEL_HIDDEN_3 = 128
KEEP_PROB_MODEL = 0.5
KEEP_PROB_AGENT = 0.5
LR = 1e-2
GAMMA = 0.99
INPUT_DIMS = 4
STATE_DIMS = 5
NUM_ACTIONS = 2
# Define training parameters
TOTAL_EPS = 5000
MAX_STEPS = 300
REAL_BSIZE = 3
MODEL_BSIZE = 3

BINARY_OBJECTIVE = True


class PolicyNet(object):
    """ Policy net encoding the agent and learning the optimal policy through interaction with the model net.
    Two objective functions are implemented: binary_objective=True, for when the agent has to decide between two actions
    and binary_objective=False, for when the action space is larger. The latter option also works for binary decision,
    yet the former offers more reliable convergence. """

    def __init__(self, input_dims, hidden_1, hidden_2, hidden_3, num_actions, learning_rate, binary_objective=True):
        self.input_dims = input_dims
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2
        self.hidden_3 = hidden_3
        self.learning_rate = learning_rate
        self.dtype = tf.float32
        self.binary = binary_objective

        if self.binary:
            self.num_actions = num_actions - 1
        else:
            self.num_actions = num_actions

        self.state = tf.placeholder(shape=[None, self.input_dims], dtype=self.dtype, name='current_state')

        if self.binary:
            self.action_holder = tf.placeholder(shape=[None, 1], dtype=self.dtype, name='actions')
        else:
            self.action_holder = tf.placeholder(shape=[None, 1], dtype=tf.int32, name='actions')
        self.reward_holder = tf.placeholder(dtype=self.dtype, name='rewards')
        self.keep_prob = tf.placeholder(dtype=self.dtype, name='keep_prob')

        with tf.variable_scope('layer_1'):
            w1 = tf.get_variable(name='weight', shape=[self.input_dims, self.hidden_1], dtype=self.dtype,
                                 initializer=xi())
            o1 = tf.nn.relu(tf.matmul(self.state, w1), name='output')
            d1 = tf.nn.dropout(o1, self.keep_prob)
        with tf.variable_scope('layer_2'):
            w2 = tf.get_variable(name='weight', shape=[self.hidden_1, self.hidden_2], dtype=self.dtype,
                                 initializer=xi())
            o2 = tf.nn.relu(tf.matmul(d1, w2), name='output')
            d2 = tf.nn.dropout(o2, self.keep_prob)
        with tf.variable_scope('layer_3'):
            w3 = tf.get_variable(name='weight', shape=[self.hidden_2, self.hidden_3], dtype=self.dtype,
                                 initializer=xi())
            o3 = tf.nn.relu(tf.matmul(d2, w3), name='hidden_1')
        with tf.variable_scope('layer_4'):
            w4 = tf.get_variable(name='weight', shape=[self.hidden_3, self.num_actions], dtype=self.dtype,
                                 initializer=xi())
            score = tf.matmul(o3, w4, name='score')
            if self.binary:
                self.probability = tf.nn.sigmoid(score, name='action_probability')
            else:
                self.probability = tf.nn.softmax(score, name='action_probabilities')

        self.t_vars = tf.trainable_variables()

        with tf.variable_scope('loss'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.gradient_holders = list()
            for _idx, var in enumerate(self.t_vars):
                placeholder = tf.placeholder(dtype=tf.float32, name=str(_idx) + '_holder')
                self.gradient_holders.append(placeholder)

            if self.binary:
                self.action_holder = tf.abs(self.action_holder - 1)
                log_lh = tf.log(
                    self.action_holder * (self.action_holder - self.probability) + (1 - self.action_holder) *
                    (self.action_holder + self.probability))
                self.loss = - tf.reduce_mean(log_lh * self.reward_holder)
            else:
                indices = tf.range(0, tf.shape(self.probability)[0]) * tf.shape(self.probability)[1] + \
                          self.action_holder
                responsible_outputs = tf.gather(tf.reshape(self.probability, [-1]), indices)
                self.loss = - tf.reduce_mean(tf.multiply(tf.log(responsible_outputs), self.reward_holder), name='loss')

            self.get_gradients = tf.gradients(self.loss, self.t_vars)
            self.batch_update = optimizer.apply_gradients(zip(self.gradient_holders, self.t_vars))


class ModelNet(object):
    """ Network predicting environment data based on previous observations. """

    def __init__(self, hidden_1, hidden_2, hidden_3, input_dims, state_dims, learning_rate):
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2
        self.hidden_3 = hidden_3
        self.input_dims = input_dims
        self.state_dims = state_dims
        self.learning_rate = learning_rate
        self.dtype = tf.float32

        self.previous_state = tf.placeholder(shape=[None, self.state_dims], dtype=self.dtype, name='model_input')
        self.true_observation = tf.placeholder(shape=[None, self.input_dims], dtype=self.dtype, name='true_obs')
        self.true_reward = tf.placeholder(shape=[None, 1], dtype=self.dtype, name='true_reward')
        self.true_done = tf.placeholder(shape=[None, 1], dtype=self.dtype, name='true_done')
        self.keep_prob = tf.placeholder(dtype=self.dtype, name='keep_prob')

        # Define layers
        with tf.variable_scope('layer_1'):
            w_1 = tf.get_variable(name='weights', shape=[self.state_dims, self.hidden_1], dtype=self.dtype,
                                  initializer=xi())
            b_1 = tf.get_variable(name='biases', shape=[self.hidden_1], dtype=self.dtype,
                                  initializer=tf.constant_initializer(0.0))
            o_1 = tf.nn.relu(tf.nn.xw_plus_b(self.previous_state, w_1, b_1), name='output')
            d_1 = tf.nn.dropout(o_1, keep_prob=self.keep_prob)
        with tf.variable_scope('layer_2'):
            w_2 = tf.get_variable(name='weights', shape=[self.hidden_1, self.hidden_2], dtype=self.dtype,
                                  initializer=xi())
            b_2 = tf.get_variable(name='biases', shape=[self.hidden_2], dtype=self.dtype,
                                  initializer=tf.constant_initializer(0.0))
            o_2 = tf.nn.relu(tf.nn.xw_plus_b(d_1, w_2, b_2), name='output')
            d_2 = tf.nn.dropout(o_2, self.keep_prob)
        with tf.variable_scope('layer_3'):
            w_3 = tf.get_variable(name='weights', shape=[self.hidden_2, self.hidden_3], dtype=self.dtype,
                                  initializer=xi())
            b_3 = tf.get_variable(name='biases', shape=[self.hidden_3], dtype=self.dtype,
                                  initializer=tf.constant_initializer(0.0))
            o_3 = tf.nn.relu(tf.nn.xw_plus_b(d_2, w_3, b_3), name='output')
        with tf.variable_scope('prediction_layer'):
            w_obs = tf.get_variable(name='state_weight', shape=[self.hidden_3, self.input_dims], dtype=self.dtype,
                                    initializer=xi())
            b_obs = tf.get_variable(name='state_bias', shape=[self.input_dims], dtype=self.dtype,
                                    initializer=tf.constant_initializer(0.0))
            w_reward = tf.get_variable(name='reward_weight', shape=[self.hidden_3, 1], dtype=self.dtype,
                                       initializer=xi())
            b_reward = tf.get_variable(name='reward_bias', shape=[1], dtype=self.dtype,
                                       initializer=tf.constant_initializer(0.0))
            w_done = tf.get_variable(name='done_weight', shape=[self.hidden_3, 1], dtype=self.dtype,
                                     initializer=xi())
            b_done = tf.get_variable(name='done_bias', shape=[1], dtype=self.dtype,
                                     initializer=tf.constant_initializer(1.0))
            predicted_observation = tf.nn.xw_plus_b(o_3, w_obs, b_obs, name='observation_prediction')
            predicted_reward = tf.nn.xw_plus_b(o_3, w_reward, b_reward, name='reward_prediction')
            predicted_done = tf.nn.sigmoid(tf.nn.xw_plus_b(o_3, w_done, b_done, name='done_prediction'))
            self.predicted_state = tf.concat(values=[predicted_observation, predicted_reward, predicted_done],
                                             axis=1, name='state_prediction')

        # Get losses
        with tf.variable_scope('loss'):
            observation_loss = tf.square(tf.subtract(self.true_observation, predicted_observation),
                                         name='observation_loss')
            reward_loss = tf.square(tf.subtract(self.true_reward, predicted_reward), name='reward_loss')
            # Cross-entropy due to one-hot nature of the done-vector (1 if match, 0 otherwise)
            done_loss = tf.multiply(self.true_done, predicted_done) + tf.multiply(1 - self.true_done,
                                                                                  1 - predicted_done)
            done_loss = - tf.log(done_loss)
            self.loss = tf.reduce_mean(1.0 * observation_loss + 1.0 * reward_loss + 2.0 * done_loss,
                                       name='combined_loss')
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.update_model = optimizer.minimize(loss=self.loss)


# Declare any helper functions
def reset_grad_buff(grad_buff):
    """ Resets the gradients kept within the gradient buffer. """
    for index, gradient in enumerate(grad_buff):
        grad_buff[index] = gradient * 0
    return grad_buff


def discount_rewards(reward_vector):
    """ Produces a discounter rewards 1D vector from the rewards collected over the duration of an episode. """
    discounted = np.zeros_like(reward_vector)
    running_add = 0
    for i in reversed(range(0, reward_vector.size)):
        running_add = running_add * GAMMA + reward_vector[i]
        discounted[i] = running_add
    discounted -= np.mean(discounted)
    discounted /= np.std(discounted)
    return discounted


def model_step_function(this_sess, model, checked_state, checked_action, current_step):
    """ Performs a single training step using the model network. """
    feed_input = np.hstack([checked_state, np.reshape(np.array(checked_action), [-1, 1])])
    # Obtain prediction
    msf_prediction = this_sess.run(model.predicted_state, feed_dict={model.previous_state: feed_input,
                                                                     model.keep_prob: 1.0})
    next_observation = msf_prediction[:, 0:4]
    next_reward = msf_prediction[:, 4]
    # Clip values
    next_observation[:, 0] = np.clip(next_observation[:, 0], - 2.4, 2.4)
    next_observation[:, 2] = np.clip(next_observation[:, 2], - 0.4, 0.4)
    done_prob = np.clip(msf_prediction[:, 5], 0, 1)
    # Check if episode done or maximum number of steps is exceeded
    if done_prob > 0.1 or current_step > MAX_STEPS:
        next_done = True
    else:
        next_done = False
    return next_observation, next_reward, next_done


total_reward = list()
episodic_reward = 0
episode_history = list()
episode_number = 0
real_episodes = 0
batch_size = REAL_BSIZE
solved = False

draw_from_model = False
train_the_model = True
train_the_policy = False

agent = PolicyNet(INPUT_DIMS, AGENT_HIDDEN_1, AGENT_HIDDEN_2, AGENT_HIDDEN_3, NUM_ACTIONS, LR)
env_model = ModelNet(MODEL_HIDDEN_1, MODEL_HIDDEN_2, MODEL_HIDDEN_3, INPUT_DIMS, STATE_DIMS, LR)

# Launch the graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    grad_buffer = sess.run(agent.t_vars)
    grad_buffer = reset_grad_buff(grad_buffer)
    new_state = env.reset()
    state = new_state

    while episode_number <= TOTAL_EPS:
        state = np.reshape(new_state, [1, 4])

        if BINARY_OBJECTIVE:
            action_prob = sess.run(agent.probability, feed_dict={agent.state: state, agent.keep_prob: 1.0})
            action = 1 if np.random.uniform() < action_prob else 0

        else:
            action_distribution = sess.run(agent.probability, feed_dict={agent.state: state, agent.keep_prob: 1.0})
            action_value = np.random.choice(action_distribution[0], p=action_distribution[0])
            match = np.square(action_distribution - action_value)
            action = np.argmin(match)

        # Perform a single step either within the model or the real environment to obtain new measurements
        if draw_from_model:
            new_state, reward, done = model_step_function(sess, env_model, state, action, len(episode_history))
        else:
            new_state, reward, done, info = env.step(action)

        episode_history.append([state, action, reward, done, new_state])
        episodic_reward += reward

        if done:
            if not draw_from_model:
                real_episodes += 1
                total_reward.append(episodic_reward)
            episode_number += 1
            episodic_reward = 0

            episode_history = np.array(episode_history)
            # Unravel the history
            episode_state = np.vstack(episode_history[:, 0])
            episode_action = np.reshape(episode_history[:, 1], [-1, 1])
            episode_reward = np.reshape(episode_history[:, 2], [-1, 1])
            episode_done = np.reshape(episode_history[:, 3], [-1, 1])
            episode_next = np.vstack(episode_history[:, 4])
            # episode_check = np.reshape(episode_history[:, 5], [-1, 1])
            episode_history = list()
            # Train each of the networks when specified
            if train_the_model:
                state_plus_action = np.hstack([episode_state, episode_action])
                episode_all = np.hstack([episode_next, episode_reward, episode_done])
                feed_dict = {env_model.previous_state: state_plus_action,
                             env_model.true_observation: episode_next,
                             env_model.true_done: episode_done,
                             env_model.true_reward: episode_reward,
                             env_model.keep_prob: KEEP_PROB_MODEL}
                loss, state_prediction, _ = sess.run([env_model.loss, env_model.predicted_state,
                                                      env_model.update_model], feed_dict=feed_dict)

            if train_the_policy:
                discounted_reward = discount_rewards(episode_reward).astype('float32')
                feed_dict = {agent.state: episode_state,
                             agent.action_holder: episode_action,
                             agent.reward_holder: discounted_reward,
                             agent.keep_prob: KEEP_PROB_AGENT}
                agent_gradients = sess.run(agent.get_gradients, feed_dict=feed_dict)
                # Break if gradients become too large
                if np.sum(agent_gradients[0] == agent_gradients[0]) == 0:
                    break
                for idx, grad in enumerate(agent_gradients):
                    grad_buffer[idx] += grad

            if episode_number % batch_size == 0 and real_episodes >= 100:
                if train_the_policy:
                    _ = sess.run(agent.batch_update, feed_dict=dict(zip(agent.gradient_holders, grad_buffer)))
                    grad_buffer = reset_grad_buff(grad_buffer)

                if not draw_from_model:
                    batch_reward = np.mean(total_reward[- REAL_BSIZE:])
                    mean_total = np.mean(total_reward[- REAL_BSIZE * 100:])
                    print('Acting in env. | Episode: %d | Batch reward %.4f | Action: %.4f | Mean reward: %.4f'
                          % (real_episodes, batch_reward, action, mean_total))
                    if batch_reward >= 200:
                        solved = True

                # Once the model has been trained on 100 episodes, we start alternating between training the policy
                # from the model and training the model from the real environment.
                if episode_number > 100:
                    draw_from_model = not draw_from_model
                    train_the_model = not train_the_model
                    train_the_policy = not train_the_policy

            if draw_from_model:
                new_state = np.random.uniform(-0.1, 0.1, [4])  # Generate reasonable starting point
                batch_size = MODEL_BSIZE
            else:
                new_state = env.reset()
                batch_size = REAL_BSIZE

            if episode_number % 1000 == 0:
                LR /= 2

            if solved:
                print('Found a solution!')
                break

print('Agent has experienced %d real episodes.' % real_episodes)
