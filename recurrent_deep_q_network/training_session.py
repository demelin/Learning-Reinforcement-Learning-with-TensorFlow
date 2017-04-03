import os
import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import LSTMCell
from q_learning.q_network import MentorAgent, ExperienceBuffer, update_target_graph, perform_update, process_capture

import gym
import universe

tf.reset_default_graph()

env = gym.make('Pong-v0')

# Network constants
FILTER_DIMS = [[8, 8], [4, 4], [3, 3], [6, 6]]
FILTER_NUMS = [32, 64, 64, 512]
STRIDES = [[4, 4], [2, 2], [1, 1], [1, 1]]
HIDDEN_SIZE = 512
ACTION_NUM = 6  # According to documentation
LEARNING_RATE = 1e-4
BUFFER_SIZE = 1000

# Session constants
BATCH_SIZE = 4
TRACE_LENGTH = 8
UPDATE_FREQ = 5
TAU = 0.99  # Discount factor on target Q-values
START_RAND = 1.0
END_RAND = 0.1
ANN_STEPS = 10000
NUM_EPISODES = 10000
PRE_TRAIN_STEPS = 10000
LOAD_MODEL = False
PATH = os.curdir + '/rdqn/model'
MAX_EPISODE_LENGTH = 50
SUMMARY_LENGTH = 100
SAVING_FREQ = 10000

# Defines cells to be used in the actor and the target network
actor_cell = LSTMCell(num_units=HIDDEN_SIZE, state_is_tuple=True)
target_cell = LSTMCell(num_units=HIDDEN_SIZE, state_is_tuple=True)
# Initialize networks and buffer
actor_qn = MentorAgent(HIDDEN_SIZE, actor_cell, FILTER_DIMS, FILTER_NUMS, STRIDES, 'actor', ACTION_NUM, LEARNING_RATE)
target_qn = \
    MentorAgent(HIDDEN_SIZE, target_cell, FILTER_DIMS, FILTER_NUMS, STRIDES, 'target', ACTION_NUM, LEARNING_RATE)
session_buffer = ExperienceBuffer(BUFFER_SIZE)

# Define target_qn update OPs to be used in the session (tf.trainable_variables() operates on the graph)
tvars = tf.trainable_variables()
actor_tvars, target_tvars = tvars[:len(tvars)//2], tvars[len(tvars)//2:]
target_ops = update_target_graph(actor_tvars, target_tvars, TAU)

saver = tf.train.Saver(max_to_keep=5)

# Scheduling e-greedy exploration
epsilon = START_RAND
drop_per_step = (START_RAND - END_RAND) / ANN_STEPS

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

    # Set target network equal to the agent network
    perform_update(target_ops, sess)
    # Manage summaries
    merged = tf.summary.merge_all()
    training_writer = tf.summary.FileWriter('./train', sess.graph)

    # Enter training loop
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

        # Enter the Q-Network loop (play until a single game is completed, alternatively uncomment for max_ep_len)
        # while step < MAX_EPISODE_LENGTH:
        while True:
            # step += 1
            feed_dict = {actor_qn.scalar_input: proc_env_state, actor_qn.trace_length: 1,
                         actor_qn.state_in: rnn_state, actor_qn.batch_size: 1}
            # Choose action following the e-greedy strategy
            if np.random.rand(1) < epsilon or total_steps < PRE_TRAIN_STEPS:
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
            # Proceed with exploitation once the exploration phase is concluded
            if total_steps > PRE_TRAIN_STEPS:
                if epsilon > END_RAND:
                    epsilon -= drop_per_step

                # Update target network
                if total_steps % (UPDATE_FREQ * 1000) == 0:
                    perform_update(target_ops, sess)

                # Update agent network
                if total_steps % UPDATE_FREQ == 0:
                    # Reset the RNN hidden state
                    rnn_state_train = (np.zeros([BATCH_SIZE, HIDDEN_SIZE]), np.zeros([BATCH_SIZE, HIDDEN_SIZE]))
                    # Get random batch of experiences from the experience buffer
                    train_batch = session_buffer.sample_experience(BATCH_SIZE, TRACE_LENGTH)

                    # Perform the Double-DQN update to the target Q-values
                    # Agent network
                    q_1 = sess.run(actor_qn.prediction,
                                   feed_dict={actor_qn.scalar_input: (np.vstack(train_batch[:, 3]) / 255.0),
                                              actor_qn.trace_length: TRACE_LENGTH,
                                              actor_qn.state_in: rnn_state_train,
                                              actor_qn.batch_size: BATCH_SIZE})
                    # Target network
                    q_2 = sess.run(target_qn.q_out,
                                   feed_dict={target_qn.scalar_input: (np.vstack(train_batch[:, 3]) / 255.0),
                                              target_qn.trace_length: TRACE_LENGTH,
                                              target_qn.state_in: rnn_state_train,
                                              target_qn.batch_size: BATCH_SIZE})
                    # Exclude final steps in each episode
                    end_multiplier = np.abs(train_batch[:, 4] - 1)
                    # Select q-values from target network based on actions predicted by the agent network
                    double_q = q_2[range(BATCH_SIZE * TRACE_LENGTH), q_1]
                    # See traget-Q double-DQN update equation
                    target_q = train_batch[:, 2] + (TAU * double_q * end_multiplier)
                    # Update agent network with the so obtained target_q values
                    _ = sess.run(actor_qn.update_model,
                                 feed_dict={actor_qn.scalar_input: (np.vstack(train_batch[:, 0]) / 255.0),
                                            actor_qn.target_q_holder: target_q,
                                            actor_qn.action_holder: train_batch[:, 1],
                                            actor_qn.trace_length: TRACE_LENGTH,
                                            actor_qn.state_in: rnn_state_train,
                                            actor_qn.batch_size: BATCH_SIZE})

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
        # episode_buffer = zip(buffer_array)
        session_buffer.add_experience(buffer_array, TRACE_LENGTH, buffer_array.shape[0])
        # Update tracking lists
        steps_per_episode.append(step)
        total_rewards.append(running_reward)

        # Save model periodically
        if i % SAVING_FREQ == 0 and i != 0:
            saver.save(sess, PATH + '/model-' + str(i) + '.cptk')
            print('Model saved after %d steps!' % i)
        # Report on the training performance of the actor network
        if i % SUMMARY_LENGTH == 0 and i != 0:
            print('Episode: %d | Steps taken: %d | Average episodic reward: %.4f | epsilon value: %.4f'
                  % (i, total_steps, np.mean(total_rewards[-SUMMARY_LENGTH:]), epsilon))

    # Save final model
    saver.save(sess, PATH + '/model-final' + '.cptk')
