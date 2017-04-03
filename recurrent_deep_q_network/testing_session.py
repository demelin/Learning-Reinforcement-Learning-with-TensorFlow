import os
import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import LSTMCell
from q_learning.q_network import MentorAgent, process_capture

import gym
import universe

tf.reset_default_graph()

env = gym.make('Pong-v0')

# Network constants
FILTER_DIMS = [[8, 8], [4, 4], [3, 3], [6, 6]]
FILTER_NUMS = [32, 64, 64, 512]
STRIDES = [[4, 4], [2, 2], [1, 1], [1, 1]]
HIDDEN_SIZE = 512
ACTION_NUM = 6
LEARNING_RATE = 1e-4
BUFFER_SIZE = 1000

# Session constants
EPSILON = 0.01
NUM_EPISODES = 10000
LOAD_MODEL = True
PATH = os.curdir + '/rdqn/model'
# MAX_EPISODE_LENGTH = 50
SUMMARY_LENGTH = 10
SAVING_FREQ = 1000

# Defines cells to be used in the actor and the target network
actor_cell = LSTMCell(num_units=HIDDEN_SIZE, state_is_tuple=True)
# Initialize networks and buffer
actor_qn = MentorAgent(HIDDEN_SIZE, actor_cell, FILTER_DIMS, FILTER_NUMS, STRIDES, 'actor', ACTION_NUM, LEARNING_RATE)

saver = tf.train.Saver(max_to_keep=2)

# Initialize tracking variables
steps_per_episode = list()
total_rewards = list()
total_steps = 0

# Make path for model saving
if not os.path.exists(PATH):
    os.makedirs(PATH)

# Start the session
with tf.Session() as sess:
    if LOAD_MODEL:
        print('Loading model ... ')
        checkpoint = tf.train.get_checkpoint_state(PATH)
        saver.restore(sess, checkpoint.model_checkpoint_path)

    sess.run(tf.global_variables_initializer())

    # Enter the testing loop
    for i in range(NUM_EPISODES):
        # Keep track of episodes and steps completed
        print('Episode %d | Total steps taken: %d' % (i, total_steps))
        episode_buffer = list()
        # Get new observations
        env_state = env.reset()
        proc_env_state = process_capture(env_state)
        done = False
        running_reward = 0
        step = 0
        # Reset RNN hidden state
        rnn_state = (np.zeros([1, HIDDEN_SIZE]), np.zeros([1, HIDDEN_SIZE]))

        # Enter the Q-Network loop
        while True:
            step += 1
            feed_dict = {actor_qn.scalar_input: proc_env_state, actor_qn.trace_length: 1,
                         actor_qn.state_in: rnn_state, actor_qn.batch_size: 1}
            # Choose action following the e-greedy strategy
            if np.random.rand(1) < EPSILON:
                # Take a random action
                rnn_state_1 = sess.run(actor_qn.final_state, feed_dict=feed_dict)
                action = np.random.randint(0, 3)
            else:
                # Obtain action from model
                action, rnn_state_1 = sess.run([actor_qn.prediction, actor_qn.final_state], feed_dict=feed_dict)
                action = action[0]
            # Take a step in the environment
            env_state_1, reward, done, _ = env.step(action)
            proc_env_state_1 = process_capture(env_state_1)
            total_steps += 1
            # Add interaction to the episode buffer
            episode_buffer.append(np.reshape([proc_env_state, action, reward, proc_env_state_1, done], [1, 5]))
            # Update environment interaction variables
            running_reward += reward
            proc_env_state = proc_env_state_1
            env_state = env_state_1
            rnn_state = rnn_state_1
            # Terminate episode once done
            if done:
                break

        # Add episode to the experience buffer
        buffer_array = np.array(episode_buffer)
        # Update tracking lists
        steps_per_episode.append(step)
        total_rewards.append(running_reward)

        # Report on the model's performance
        if i % SUMMARY_LENGTH == 0 and i != 0:
            print('Steps taken: %d | Average episodic reward: %.4f'
                  % (total_steps, np.mean(total_rewards[-SUMMARY_LENGTH:])))

    # Report on final performance of the model
    print('Percent of successfully completed games: %.4f' % (sum(total_rewards) / NUM_EPISODES))
