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
random_threshold = 1000

def new_weights(shape, name="weight"):
    '''Creates and initializes weight matrices'''
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)

def new_biases(length, name="bias"):
    '''Creates bias vectors for layers'''
    return tf.Variable(tf.constant(0.1, shape=[length]), name=name)

def new_layer(input, num_inputs, num_outputs, use_relu=True, name="new_layer"):
    '''Creates a new dense layer with dimensions as per the parameters'''
    weights = new_weights(shape=[num_inputs, num_outputs], name="{}_weights".format(name))
    biases = new_biases(length=num_outputs, name="{}_biases".format(name))
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

class QNet():
    def __init__(self, trainable=False, name="DQN"):
        self.env_obs = tf.placeholder(shape=[None, env_features], dtype=tf.float32)
        self.fc_1 = new_layer(self.env_obs, env_features, 128, name="{}_fc_1".format(name))
        self.fc_2 = new_layer(self.fc_1, 128, 512, name="{}_fc_2".format(name))
        self.fc_3 = new_layer(self.fc_2, 512, 512, name="{}_fc_3".format(name))
        self.fc_4 = new_layer(self.fc_3, 512, 128, name="{}_fc_4".format(name))
        self.fc_5 = new_layer(self.fc_4, 128, env.action_space.n, use_relu=False, name="{}_fc_1".format(name))
        self.q_out = tf.nn.softmax(self.fc_5, name="{}_q_out".format(name))
        self.predict = tf.argmax(self.q_out)
        if trainable:
            self.q_next = tf.placeholder(shape=[None, env.action_space.n], dtype=tf.float32, name="{}_q_next".format(name))
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_next, self.q_out))
            tf.summary.scalar('loss_function', self.loss)
            self.train_op = tf.train.RMSPropOptimizer(0.05).minimize(self.loss)

mainQN = QNet(trainable=True, name="mainQN")
targetQN = QNet(name="targetQN")

with tf.Session() as sess:
    frames = 0
    saver = tf.train.Saver()
    merged = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("train_summary", sess.graph)
    f_reward = 0
    for i in range(num_episodes):
        #Reset environment and get first new observation
        state = env.reset()
        done = False
        while not done:
            env.render()
            frames += 1
            if frames < random_threshold or np.random.rand(1) < e:
                action = env.action_space.sample()
            else:
                action = sess.run([mainQN.predict],feed_dict={mainQN.env_obs:[state]})
                action = np.argmax(action)
            #Get new state and reward from environment
            observation,reward,done,_ = env.step(action)
            store_transition(state, action, reward, observation)
            if frames >= random_threshold and frames % 4 == 0:
                if table_idx > memory_size:
                    sample_index = np.random.choice(memory_size, size=train_batch_size)
                else:
                    sample_index = np.random.choice(table_idx, size=train_batch_size)
                batch = table[sample_index, :]
                _observations = batch[:, :env_features]
                _observations_next = batch[:, -env_features:]
                q_next = sess.run(targetQN.q_out, feed_dict={targetQN.env_obs: _observations_next})
                q_eval_next = sess.run(mainQN.q_out, feed_dict={mainQN.env_obs: _observations_next})
                q_eval = sess.run(mainQN.q_out, feed_dict={mainQN.env_obs: _observations})
                q_target = q_eval.copy()
                batch_index = np.arange(train_batch_size, dtype=np.int32)
                eval_act_index = batch[:, env_features].astype(int)
                _reward = batch[:, env_features + 1]
                max_action = np.argmax(q_eval_next, axis=1)
                next_selected = q_next[batch_index, max_action]
                q_target[batch_index, eval_act_index] = _reward + y * next_selected
                sess.run(mainQN.train_op, feed_dict={mainQN.env_obs: _observations, mainQN.q_next: q_target})
                if done:
                    summary = sess.run(merged, feed_dict={mainQN.env_obs:_observations, targetQN.env_obs:_observations, mainQN.q_next: q_target})
                    writer.add_summary(summary, i)
            state = observation
            f_reward += reward
            if done:
                saver.save(sess, 'mspacman/model')
                print('mean_reward: ',f_reward/(i+1))
