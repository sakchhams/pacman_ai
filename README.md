# pacman_ai
Deep Q Network optimized for mspacman environment(OpenAI-gym)

## [MsPacman-ram-v0](https://gym.openai.com/envs/MsPacman-ram-v0/)
This environment provides the ram(128 bytes!) of tha atari console as input.

The script records and stores each transition of the environment, that is (state, action, reward, observation) which is later used to train the model to imporve predictions. It uses the [Bellman Eqation](https://en.wikipedia.org/wiki/Bellman_equation) to estimate the next values of the `Q-Table` wich is used in computing a loss function(RMS) for optimization to improve the efficiency of the network.

Network after a few hundred iterations is able to achive an average score of 400-500.

## [MsPacman-v0](https://gym.openai.com/envs/MsPacman-v0/)
This environment provides the current display state as input, a tensor with dimensons (210, 160, 3).

The environment is first passed to a 3 layered convolution network to extract features which is then fed to a Deep Q-Network to estimate the Q values. This Q-Network works almost the same as the above network as in case of MsPacman-ram with the difference that the sates, action, reward and observation are not recorded.

Due to computational limitations this network has not been trained for enough iterations to get a meaningful result. 

The above networks can be used for all atari environments with very minor changes.
