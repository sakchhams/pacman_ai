"""
Solving MsPacman-ram-v0 environment
Author: Sakchham Sharma(sakchhams@gmail.com)
"""

import gym
import numpy as np
import tensorflow as tf

#learning parameters
y = .95 #gamma
e = 0.1 #random selection epsilion
memory_size = 1000
num_episodes = 2000
env_features = 128 #128 bytes of atari console's RAM
train_batch_size = 64

def new_weights(shape):
    '''Creates and initializes weight matrices'''
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def new_biases(length):
    '''Creates bias vectors for layers'''
    return tf.Variable(tf.constant(0.1, shape=[length]))

def new_layer(input, num_inputs, num_outputs, use_relu=True):
    '''Creates a new dense layer with dimensions as per the parameters'''
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer

env = gym.make('MsPacman-ram-v0')

#build a table to store previous game states
table = np.zeros((memory_size, env_features * 2 + 2))

def store_transition(state, action, reward, observation):
    '''stores a transition for training the model'''
    if 'table_idx' not in globals():
        global table_idx
        table_idx = 0
    transition = np.hstack((state, [action, reward], observation))
    index = table_idx % memory_size #overwrite old values
    table[index, :] = transition
    table_idx += 1

#value evaluation Q-network
with tf.variable_scope('q_net'):
    env_obs = tf.placeholder(shape=[None,env_features], dtype=tf.float32, name='state')
    with tf.variable_scope('hidden_layer_1'):
        h_layer_1 = new_layer(env_obs, env_features, 64)
    with tf.variable_scope('hidden_layer_2'):
        h_layer_2 = new_layer(h_layer_1, 64, 32)
    with tf.variable_scope('output_layer'):
        o_layer = new_layer(h_layer_2, 32, env.action_space.n, False)
    q_out = tf.nn.softmax(o_layer, name='q_out')
    predict = tf.argmax(q_out, name='action')

    q_next = tf.placeholder(shape=[None,env.action_space.n],dtype=tf.float32,name='q_next')
    loss = tf.reduce_mean(tf.squared_difference(q_next, q_out), name='loss')
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.RMSPropOptimizer(0.05).minimize(loss)

with tf.Session() as sess:
    saver = tf.train.Saver()
    train_writer = tf.summary.FileWriter('mspacman-ram_summary/',sess.graph)
    sess.run(tf.global_variables_initializer())
    f_reward = 0
    for i in range(num_episodes):
        #Reset environment and get first new observation
        state = env.reset()
        done = False
        while not done:
            env.render()
            action, Q = sess.run([predict, q_out],feed_dict={env_obs:[state]})
            action = np.argmax(action)
            if np.random.rand(1) < e:
                action = env.action_space.sample()
            #Get new state and reward from environment
            observation,reward,done,_ = env.step(action)
            store_transition(state, action, reward, observation)
            if i < 10:
                Q1 = sess.run(q_out, feed_dict={env_obs: [observation]})
                q_target = Q
                q_target[0,action] = reward + y*np.max(Q1)
            else:
                if table_idx > memory_size:
                    sample_index = np.random.choice(memory_size, size=train_batch_size)
                else:
                    sample_index = np.random.choice(table_idx, size=train_batch_size)
                batch = table[sample_index, :]
                #Q1, Q2 = sess.run([q_out, q_target_next], feed_dict={env_obs:batch[:, :env_features], next_state:batch[:, -env_features:]})
                Q1 = sess.run(q_out, feed_dict={env_obs:batch[:, :env_features]})
                Q2 = sess.run(q_out, feed_dict={env_obs:batch[:, -env_features:]})
                q_target = Q1.copy()
                batch_index = np.arange(train_batch_size, dtype=np.int32)
                eval_act_index = batch[:, env_features].astype(int)
                _reward = batch[:, env_features + 1]
                q_target[batch_index, eval_act_index] = _reward + y * np.max(Q2, axis=1)
            sess.run(optimizer,feed_dict={env_obs:[state],q_next:q_target})
            state = observation
            f_reward += reward
            if done:
                saver.save(sess, 'mspacman-ram/model')
                merged = sess.run(tf.summary.merge_all(), feed_dict={env_obs:[state], q_next:q_target})
                train_writer.add_summary(merged, i)
                print('mean_reward: ',f_reward/(i+1))
