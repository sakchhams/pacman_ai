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

with tf.variable_scope('q_net'):
    env_obs = tf.placeholder(shape=[None, 210, 160, 3], dtype=tf.float32, name='env_obs')
    with tf.variable_scope('l1'):
        layer1 = new_conv_layer(input=env_obs, num_input_channels=3,filter_size=5,num_filters=32,max_pooled=True)
    with tf.variable_scope('l2'):
        layer2 = new_conv_layer(input=layer1, num_input_channels=32,filter_size=5,num_filters=32,max_pooled=True)
    with tf.variable_scope('l3'):
        layer3 = new_conv_layer(input=layer2, num_input_channels=32,filter_size=5,num_filters=64,max_pooled=True)
    with tf.variable_scope('fc1'):
        l_flat, num_features = flatten_layer(layer3)
        fc_1 = new_fc_layer(l_flat, num_features, 128)
    with tf.variable_scope('fc2'):
        fc_2 = new_fc_layer(fc_1, 128, env.action_space.n, False)
    q_out = tf.nn.softmax(fc_2, name='q_out')
    predict = tf.argmax(q_out, name='action')
    q_next = tf.placeholder(shape=[None,env.action_space.n],dtype=tf.float32,name='q_next')
    loss = tf.reduce_mean(tf.squared_difference(q_next, q_out), name='loss')
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.RMSPropOptimizer(0.05).minimize(loss)

with tf.Session() as sess:
    saver = tf.train.Saver()
    train_writer = tf.summary.FileWriter('mspacman_summary/',sess.graph)
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
            Q1 = sess.run(q_out, feed_dict={env_obs: [observation]})
            q_target = Q
            q_target[0,action] = reward + y*np.max(Q1)
            sess.run(optimizer,feed_dict={env_obs:[state],q_next:q_target})
            state = observation
            f_reward += reward
            if done:
                saver.save(sess, 'mspacman/model')
                merged = sess.run(tf.summary.merge_all(), feed_dict={env_obs:[state], q_next:q_target})
                train_writer.add_summary(merged, i)
                print('mean_reward: ',f_reward/(i+1))
