# CarRacing-ImitationLearning

The project is composed of six different files with different functionalities:

 - demonstrator.py is the script allowing any user to record demonstrations to be later used for composing an IL dataset.
 - Simulator.py allows for policy visualization and testing for a given model. The function computeAvgReturns enables the user to test and record the performance of multiple models in a folder and its best suited for computing and plotting the RL learning curve.
 - DeepQNetwork.py defines the CNN general architecture used in my experiments and implements some utility methods.
 - Dataset.py implements a Pytorch dataset with a number of utility functions to help transforming the data to ease the learning process.
 - training.py is the script controlling the experiments relative to the Deep Q-Learning algorithm
 - training_mc.py is the script controlling the experiments relative to the Monte Carlo algorithm

In order to perform imitation learning from scratch, a user needs first to record few demonstrations with demonstrator.py and process them with Dataset.py. It is suggested to create the dataset separately from running the actual training in order to speed up the process. Once the dataset is ready it can be used by any of the two training scripts. Before performing training the user must choose wheather running multiple processes for performing hyperparameter search or simply select a specific combination of parameters for testing a specific architecture. 
