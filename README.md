# CarRacing-ImitationLearning

## Overview
The project is composed of six different files with different functionalities:

 - `demonstrator.py` is the script allowing any user to record demonstrations to be later used for composing an Imitation Learning dataset.
 - `Simulator.py` allows for policy visualization and testing for a given model. The function `computeAvgReturns` enables the user to test and record the performance of multiple models in a folder and its best suited for computing and plotting the RL learning curve.
 - `DeepQNetwork.py` defines the CNN general architecture used in my experiments and implements some utility methods.
 - `Dataset.py` implements a Pytorch dataset with a number of utility functions to help transforming the data to ease the learning process.
 - `training.py` is the script controlling learning experiments using a simple Deep Q-Learning algorithm
 - `training_mc.py` is the script controlling learning experiments using a Monte Carlo algorithm

In order to perform imitation learning from scratch, a user needs first to record few demonstrations with `demonstrator.py` and process them with `Dataset.py`. 
It is suggested to create the dataset independently before running the actual training in order to speed up the process although the Dataset can actually be directly created during the execution of one of the training scripts. 
Once the Dataset is ready, it can be used by any of the two training scripts. 
Before performing training the user must choose whether running multiple processes for performing hyperparameter search or simply select a specific combination of parameters for testing a specific architecture. 
Finally, any saved model can be tested using `Simulator.py`.

## End-to-end Imitation Learning
In order to solve problems in the CarRacing-v0 environment from OpenAI Gym by using a set of user defined/recorded demonstrations it is necessary to follow three main steps: 

- **Record new demonstrations** 
- **Create Dataset** : Elaborate the newly recorded data to create a Dataset of demonstrations to train a Reinforcement Learning agent
- **Training** : Train an agent on the provided demonstrations

The project still presents a number of flaws which are covered in the next sections in order to ensure a successful training

### Record new demonstrations
`demonstrator.py` is the script to be used to run CarRacing-v0 and play the game manually by controlling the car (the agent) through the arrow keys.

When running the script, the user can provide a sequence of independent demonstrations in the given environment. This process can be stopped by simply closing the game window at any time.

Each single demonstation is saved in a separate file named "transitions" and followed by a demonstration id. By default the id always starts from zero (see variable `episode`) and therefore, if `demonstrator.py` is run multiple times in a row, the oldest demonstrations with the same id will be overwritten.

It is suggested to record a good number of demonstrations in order to provide the learning agent with a diverse set of examples from which it can learn. This process does not need to be run all in once but can be stopped and resumed by simply changing the variable `episode` to the desired demonstration id (it only needs to be greater than the last saved one).

At the end of the demonstration phase move all the transitions files into a folder. 

#### Suitable Demonstrations
CarRacing-v0 is a simple environment which can be solved using discrete actions which is the choice for this implementation. Because of this reason the agent will be able to learn from demonstrations where it can interpret the actions taken by the user/demonstrator. 

In the current implementation the agent will be able to interpret actions corresponding to pressing a single arrow key at a time. For instance, the agent will not be able to use demonstrations where the demonstrator decides to turn and accelerate simultaniously. In either way a transition file will be created and in case of misleading actions we will encounter an exception when running `Dataset.py`.

### Create Dataset
A demonstrations Dataset can be created only after having recorded a number of demonstrations with `demonstrator.py` and having stored them in an empty folder.

`Dataset.py` contains the definition of a `TransitionsDataset` class which deals with data loading and basic transformations to convert the transitions files into a Dataset suitable for performing training.

The file can be run as a script in order to create the Dataset in an independent step so to be used later for possibly multiple learning instances. By default, running `Dataset.py` will create a `TransitionsDataset` from files contained in a folder called "transitions" and will convert all the demonstrations into Monte Carlo demonstrations by setting the parameter `mc=True` (set `mc=False` if training with `training.py` although the algorithm is very unstable and it is not granted to converge) 

`TransitionsDataset` also receives another parameter `balance` which aims at balancing the number of samples containing a "no action" (do nothing action) with all the other possible actions. This is process strongly changes the meaning of the demonstrations recorded by the user and can speed up the learning process. The most successful models were tested without any balancing effect and therefore this parameter is set to `None` when running `Datase.py` as a script.

#### ValueError
Running `Dataset.py` for the first time on newly created demonstrations can encounter a ValueError due to _wrong_ demonstrations. As previously anticipated, suitable demonstrations must not include transitions where multiple actions are taken simultaneously (pressing multiple keys at once). If any of the demonstrations presents an ambiguous action the correspondent transition will need to be removed manually by the user. Unfortunately in the current implementation the name of the faulty demonstration is not printed in the error statement and the user needs to find it debugging the provided code.

### Training
The project presents two very similar scripts for training but only `training_mc.py` implements a learning algorithm obtaining reasonable results. Nevertheless, the following usage description works for both the training scripts.

The script can be used to control multiple experiments over different possible implementations a user might want to test. The variables controlling this behaviour can all be found at the beginning of the script and can simply be set for testing only one specific configuration.

To train your network on your personal dataset you need to change the dataset name during the creation of the `TransitionsDatset` object.

#### CPU vs GPU
The project was developed on "CPU only" machine while the experiments were run using Google Colab thus the `training_mc.py` presents a variable `colab` which allows the user to switch from CPU mode to GPU mode. While CPU mode works perfectly on the folder system created in the previous steps, GPU mode sets the root folder for the dataset and results to a Google Drive folder. If a user needs to run experiments using a GPU machine not on Google Colab, folder naming must be chcked.
