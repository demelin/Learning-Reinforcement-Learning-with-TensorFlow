""" An implementation of a deep, recurrent Q-Network following
https://gist.github.com/awjuliani/35d2ab3409fc818011b6519f0f1629df#file-deep-recurrent-q-network-ipynb. 
Adopted to play Pong, or at least to make an honest attempt at doing so, by taking cues from 
https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
"""

import numpy as np
import tensorflow as tf
import random

from tensorflow.contrib.layers import xavier_initializer as xi
from tensorflow.contrib.layers import xavier_initializer_conv2d as xi_2d


def process_capture(capture):
    """ General processing function, can be expanded. """
    cropped = capture[35:195]  # shape = [80, 80, 3]
    downsampled = cropped[::2, ::2]
    flattened = np.reshape(downsampled, [1, -1])  # shape = [1, 80 * 80 * 3]
    return flattened


class ExperienceBuffer(object):
    """ Collects the agent's experiences to be used in training. """

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = list()

    def add_experience(self, experience, trace_length, experience_length):
        """ Adds an experience to the buffer. """
        # Delete old experiences if buffer full
        held_after = len(self.buffer) + 1
        if held_after >= self.buffer_size:
            self.buffer[0: held_after - self.buffer_size] = []
        if experience_length >= trace_length:
            self.buffer.append(experience)

    def sample_experience(self, batch_size, trace_length):
        """ Samples experiences from the buffer for training. """
        sampled_episodes = random.sample(self.buffer, batch_size)
        sampled_traces = list()
        for e in sampled_episodes:
            starting_point = np.random.randint(0, (e.shape[0] - trace_length) + 1)
            trace_pick = e[starting_point: starting_point + trace_length]
            sampled_traces.append(trace_pick)
        sampled_traces = np.array(sampled_traces)
        sampled_traces = np.reshape(sampled_traces, [batch_size * trace_length, -1])
        return sampled_traces


def update_target_graph(actor_tvars, target_tvars, tau):
    """ Updates the variables of the target graph using the variable values from the actor, following the DDQN update
    equation. """
    op_holder = list()
    # .assign() is performed on target graph variables with discounted actor graph variable values
    for idx, variable in enumerate(target_tvars):
        op_holder.append(
            target_tvars[idx].assign(
                (variable.value() * tau) + ((1 - tau) * actor_tvars[idx].value())
            )
        )
    return op_holder


def perform_update(op_holder, sess):
    """ Executes the updates on the target network. """
    for op in op_holder:
        sess.run(op)


class MentorAgent(object):
    """ Agent network designed to learn via deep Q-learning. Both actor and mentor are instances of the same
    network class. """

    def __init__(self, hidden_size, rnn_cell, filter_dims, filter_nums, strides, all_scope, action_num, learning_rate):
        self.hidden_size = hidden_size
        self.rnn_cell = rnn_cell
        self.filter_dims = filter_dims
        self.filter_nums = filter_nums
        self.strides = strides
        self.all_scope = all_scope
        self.action_num = action_num
        self.learning_rate = learning_rate
        self.dtype = tf.float32

        # Define placeholders for input, training parameters, and training values
        self.scalar_input = tf.placeholder(shape=[None, 80 * 80 * 3], dtype=self.dtype, name='scalar_input')
        self.trace_length = tf.placeholder(dtype=tf.int32, name='train_duration')
        self.batch_size = tf.placeholder(dtype=tf.int32, name='batch_size')
        # Both below have shape=[batch_size * trace_len]
        self.target_q_holder = tf.placeholder(shape=[None], dtype=self.dtype, name='target_Q_values')
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32, name='actions_taken')

        # Reshape the scalar input into image-shape
        cnn_input = tf.reshape(self.scalar_input, shape=[-1, 80, 80, 3])

        # Filter output calculation: W1 = (W−F+2P)/S+1 72/4
        # Define ConvNet layers for screen image analysis
        with tf.variable_scope(self.all_scope + '_cnn_1'):
            w_1 = tf.get_variable(name='weight', shape=[*self.filter_dims[0], 3, self.filter_nums[0]],
                                  initializer=xi_2d())
            b_1 = tf.get_variable(name='bias', shape=[self.filter_nums[0]], initializer=tf.constant_initializer(0.1))
            c_1 = tf.nn.conv2d(cnn_input, w_1, strides=[1, *self.strides[0], 1], padding='VALID', name='convolution')
            o_1 = tf.nn.relu(tf.nn.bias_add(c_1, b_1), name='output')  # shape=[19, 19, 32]

        with tf.variable_scope(self.all_scope + '_cnn_2'):
            w_2 = tf.get_variable(name='weight', shape=[*self.filter_dims[1], self.filter_nums[0], self.filter_nums[1]],
                                  initializer=xi_2d())
            b_2 = tf.get_variable(name='bias', shape=[self.filter_nums[1]], initializer=tf.constant_initializer(0.1))
            c_2 = tf.nn.conv2d(o_1, w_2, strides=[1, *self.strides[1], 1], padding='VALID', name='convolution')
            o_2 = tf.nn.relu(tf.nn.bias_add(c_2, b_2), name='output')  # shape=[8, 8, 64]

        with tf.variable_scope(self.all_scope + '_cnn_3'):
            w_3 = tf.get_variable(name='weight', shape=[*self.filter_dims[2], self.filter_nums[1], self.filter_nums[2]],
                                  initializer=xi_2d())
            b_3 = tf.get_variable(name='bias', shape=[self.filter_nums[2]], initializer=tf.constant_initializer(0.1))
            c_3 = tf.nn.conv2d(o_2, w_3, strides=[1, *self.strides[2], 1], padding='VALID', name='convolution')
            o_3 = tf.nn.relu(tf.nn.bias_add(c_3, b_3), name='output')  # shape=[7, 7, 64]

        with tf.variable_scope(self.all_scope + '_cnn_out'):
            w_4 = tf.get_variable(name='weight', shape=[*self.filter_dims[3], self.filter_nums[2], self.filter_nums[3]],
                                  initializer=xi_2d())
            b_4 = tf.get_variable(name='bias', shape=[self.filter_nums[3]], initializer=tf.constant_initializer(0.1))
            c_4 = tf.nn.conv2d(o_3, w_4, strides=[1, *self.strides[3], 1], padding='VALID', name='convolution')
            cnn_out = tf.nn.relu(tf.nn.bias_add(c_4, b_4), name='output')  # shape=[1, 1, 512]

        # Reshape ConvNet output to [batch_size, trace_len, hidden_size] to be fed into the RNN
        cnn_flat = tf.reshape(cnn_out, shape=[-1])
        rnn_input = tf.reshape(cnn_flat, [self.batch_size, self.trace_length, self.hidden_size], name='RNN_input')

        # Initialize RNN and feed the input
        self.state_in = rnn_cell.zero_state(self.batch_size, tf.float32)
        self.rnn_outputs, self.final_state = tf.nn.dynamic_rnn(cell=self.rnn_cell, inputs=rnn_input,
                                                               initial_state=self.state_in,
                                                               scope=self.all_scope + '_rnn', dtype=self.dtype)
        # Concatenate RNN time steps
        rnn_2d = tf.reshape(self.rnn_outputs, shape=[-1, self.hidden_size])  # [batch_size * trace_len, hidden_size]

        # Split RNN output into advantage and value streams which are to guide the agent's policy
        with tf.variable_scope(self.all_scope + '_advantage_and_value'):
            a_w = tf.get_variable(name='advantage_weight', shape=[self.hidden_size / 2, self.action_num],
                                  dtype=self.dtype, initializer=xi())
            v_w = tf.get_variable(name='value_weight', shape=[self.hidden_size / 2, 1], dtype=self.dtype,
                                  initializer=xi())
            a_stream, v_stream = tf.split(rnn_2d, 2, axis=1)
            self.advantage = tf.matmul(a_stream, a_w, name='advantage')
            self.value = tf.matmul(v_stream, v_w, name='value')

        self.improve_vision = tf.gradients(self.advantage, cnn_input)

        # Predict the next action
        self.q_out = tf.add(self.value, tf.subtract(
            self.advantage, tf.reduce_mean(self.advantage, axis=1, keep_dims=True)),
                            name='predicted_action_distribution')  # shape=[batch_size * trace_len, num_actions]
        self.prediction = tf.argmax(self.q_out, axis=1, name='predicted_action')

        with tf.variable_scope(self.all_scope + 'loss'):
            # Obtain loss by measuring the difference between the prediction and the target Q-value
            actions_one_hot = tf.one_hot(self.action_holder, self.action_num, dtype=self.dtype)
            self.predicted_q = tf.reduce_sum(tf.multiply(self.q_out, actions_one_hot), axis=1,
                                             name='predicted_Q_values')
            # predicted_q and l2_loss have shape=[batch_size * trace_len] -> calculated per step
            self.l2_loss = tf.square(tf.subtract(self.predicted_q, self.target_q_holder), name='l2_loss')
            # Mask first half of the losses to only keep the 'important' values
            mask_drop = tf.zeros(shape=[self.batch_size, tf.cast(self.trace_length / 2, dtype=tf.int32)])
            mask_keep = tf.ones(shape=[self.batch_size, tf.cast(self.trace_length / 2, dtype=tf.int32)])
            mask = tf.concat([mask_drop, mask_keep], axis=1)  # shape=[batch_size, train_duration]
            flat_mask = tf.reshape(mask, [-1])
            self.loss = tf.reduce_mean(tf.multiply(self.l2_loss, flat_mask), name='total_loss')

            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.update_model = optimizer.minimize(self.loss)
