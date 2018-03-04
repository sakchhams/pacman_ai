"""
Solving MsPacman-v0 environment
Author: Sakchham Sharma(sakchhams@gmail.com)
"""

import gym
import numpy as np
import tensorflow as tf

env = gym.make('MsPacman-v0')
#learning parameters
y = .95 #gamma
e = 0.1 #random selection epsilion
num_episodes = 2000
RANDOM_THRESHOLD = 1000 #minimum number of frames to choose random actions for
memory_size = 1000
train_batch_size = 64
env_features = 210*160*3

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def new_biases(length):
    return tf.Variable(tf.constant(0.1, shape=[length]))

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def new_conv_layer(input, #the previous layer
                   num_input_channels, #channels in the previous layer
                   filter_size, #width and height of each filter
                   num_filters, #number of filters
                   max_pooled=True): #use 2x2 max-pooling
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = new_weights(shape=shape)
    biases  = new_biases(length=num_filters)
    layer = conv2d(input, weights)
    layer += biases
    if max_pooled:
        layer = max_pool_2x2(layer)
    layer = tf.nn.relu(layer)
    return layer

def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()

    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features

def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

class QNet():
    def __init__(self, trainable=False):
        self.env_obs = tf.placeholder(shape=[None, 210, 160, 3], dtype=tf.float32)
        self.layer1 = new_conv_layer(input=self.env_obs, num_input_channels=3,filter_size=5,num_filters=32,max_pooled=True)
        self.layer2 = new_conv_layer(input=self.layer1, num_input_channels=32,filter_size=5,num_filters=32,max_pooled=True)
        self.layer3 = new_conv_layer(input=self.layer2, num_input_channels=32,filter_size=5,num_filters=64,max_pooled=True)
        self.l_flat, self.num_features = flatten_layer(self.layer3)
        self.fc_1 = new_fc_layer(self.l_flat, self.num_features, 128)
        self.fc_2 = new_fc_layer(self.fc_1, 128, env.action_space.n, False)
        self.q_out = tf.nn.softmax(self.fc_2)
        self.predict = tf.argmax(self.q_out)
        if trainable:
            self.q_next = tf.placeholder(shape=[None, env.action_space.n], dtype=tf.float32)
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_next, self.q_out))
            self.train_op = tf.train.RMSPropOptimizer(0.05).minimize(self.loss)

#build a table to store previous game states
table = np.zeros((memory_size, 210 * 160 * 3 * 2 + 2))

def store_transition(state, action, reward, observation):
    '''stores a transition for training the model'''
    if 'table_idx' not in globals():
        global table_idx
        table_idx = 0
    state = np.reshape(state, env_features)
    observation = np.reshape(observation, env_features)
    transition = np.hstack((state, [action, reward], observation))
    index = table_idx % memory_size #overwrite old values
    table[index, :] = transition
    table_idx += 1

mainQN = QNet(True)
targetQN = QNet()
with tf.Session() as sess:
    frames = 0
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    f_reward = 0
    for i in range(num_episodes):
        #Reset environment and get first new observation
        state = env.reset()
        done = False
        while not done:
            env.render()
            frames += 1
            if frames < RANDOM_THRESHOLD or np.random.rand(1) < e:
                action = env.action_space.sample()
            else:
                action = sess.run([mainQN.predict],feed_dict={mainQN.env_obs:[state]})
                action = np.argmax(action)
            #Get new state and reward from environment
            observation,reward,done,_ = env.step(action)
            store_transition(state, action, reward, observation)
            if frames >= RANDOM_THRESHOLD and frames % 4 == 0:
                if table_idx > memory_size:
                    sample_index = np.random.choice(memory_size, size=train_batch_size)
                else:
                    sample_index = np.random.choice(table_idx, size=train_batch_size)
                batch = table[sample_index, :]
                _observations = batch[:, :env_features]
                _observations = np.reshape(_observations, (-1, 210, 160, 3))
                _observations_next = batch[:, -env_features:]
                _observations_next = np.reshape(_observations_next, (-1, 210, 160, 3))
                q_next = sess.run(mainQN.q_out, feed_dict={mainQN.env_obs: _observations_next})
                q_eval_next = sess.run(targetQN.q_out, feed_dict={targetQN.env_obs: _observations_next})
                q_eval = sess.run(mainQN.q_out, feed_dict={mainQN.env_obs: _observations})
                q_target = q_eval.copy()
                batch_index = np.arange(train_batch_size, dtype=np.int32)
                eval_act_index = batch[:, env_features].astype(int)
                _reward = batch[:, env_features + 1]
                max_action = np.argmax(q_eval_next, axis=1)
                next_selected = q_next[batch_index, max_action]
                q_target[batch_index, eval_act_index] = _reward + y * next_selected
                sess.run(mainQN.train_op, feed_dict={mainQN.env_obs: _observations, mainQN.q_next: q_target})
            state = observation
            f_reward += reward
            if done:
                saver.save(sess, 'mspacman/model')
                print('mean_reward: ',f_reward/(i+1))
