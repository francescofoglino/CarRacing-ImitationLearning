from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import os
from PIL import Image
import numpy as np
import random

FRAME_SKIP  = 4
FRAME_STACK = 3

GAMMA = 0.99

# Car control parameters
TURN        = 0.8
ACCELERATE  = 1.0
BREAK       = 0.4

# Image transformation to be applied to each single image in our dataset. Greyscale allows us to downscale the image
# dimensionality to only one channel while normalization helps the training process
TRSF = transforms.Compose([transforms.ToTensor(),
                           transforms.Grayscale(),
                           transforms.Normalize((0.5), (0.5))])

# Custom dataset for handling and processing the transitions to be used for imitation learning
class TransitionsDataset(Dataset):
    # @balance is the variable for controlling the number of "no action" samples in the dataset since they represent the
    # vast majority of all the actions taken. This re-balancing is supposed to improve learning performance.
    def __init__(self, datasetPath, balance=None, mc=False):
        # If we input a folder path the class will read all the transition files inside it for creating the dataset
        if os.path.isdir(datasetPath):
            transitions_files = os.listdir(datasetPath)
            data_name = "dataset_mc_" + str(len(transitions_files)) + ".npy"

            self.dataset = []
            # Process one transition file at a time
            for t_file in transitions_files:
                transit = np.load(datasetPath + t_file, allow_pickle=True).tolist()

                # Process each single transition (in the file) separately for converting the action from list to a
                # single integer in order to "communicate" with the DQN
                for t in transit:
                    # Action conversion
                    a = t["action"]
                    t["action"] = self.convertAction(a)

                    self.dataset.append(t)

                if mc:
                    self.monteCarloTarget(transit)

            if balance != None:
                self.dataset = self.balanceActionsSamples(balance)

            np.save(data_name, self.dataset)

        # If instead we input a file name it will be used as a dataset directly
        elif os.path.isfile(datasetPath):
            self.dataset = np.load(datasetPath, allow_pickle=True)
        else:
            raise ValueError("datasetPath must be either a folder to the transition files or a dataset")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    # pre-processing step necessary only during training. It transforms the images (states) accordingly to the previously
    # defined transformations
    def preProcessing(self):
        for d in self.dataset:
            s_0 = d["state_0"]
            s_1 = d["state_1"]

            d["state_0"] = self.transformState(s_0)
            d["state_1"] = self.transformState(s_1)

    # Method for transforming a single image. it is define as static since it needs to be used also by Simulator.py
    @staticmethod
    def transformState(s):
        all_s = []

        num_frames = len(s)

        for f in range(num_frames):
            all_s.append(TRSF(s[f].copy()))

        return torch.cat(all_s)

    # This is a simple function to convert the action list coming from the dataset composed of three values into an integer
    # value that can directly be used by to select the right Q-value from the DeepQNetwork.
    def convertAction(self, action):

        action = action.tolist()

        # no action
        if action == [0., 0., 0.]:
            return 0
        # turn right
        elif action == [TURN/FRAME_SKIP, 0., 0.]:
            return 1
        # turn left
        elif action == [-TURN/FRAME_SKIP, 0., 0.]:
            return 2
        # accelerate
        elif action == [0., ACCELERATE/FRAME_SKIP, 0.]:
            return 3
        # break
        elif action == [0., 0., BREAK/FRAME_SKIP]:
            return 4
        else:
            raise ValueError("Action not identified. This is probably due to a demonstration with multiple actions overlapping during a time stamp.")

    # This function balances the number of samples per action. Basically the vast majority of the actions is "no action"
    # due to the nature of the problem which can obstaculate learning.
    def balanceActionsSamples(self, b):

        transitions = self.dataset

        balance = b

        a0 = []
        a1 = []
        a2 = []
        a3 = []
        a4 = []

        for t in transitions:
            action = t["action"]

            if action == 0:
                a0.append(t)
            elif action == 1:
                a1.append(t)
            elif action == 2:
                a2.append(t)
            elif action == 3:
                a3.append(t)
            elif action == 4:
                a4.append(t)
            else:
                raise ValueError

        num_no_action       = len(a0)
        num_other_actions   = len(a1) + len(a2) + len(a3) + len(a4)

        a0_downsampled = []

        if num_no_action > num_other_actions*balance:
            a0_downsampled = random.sample(a0, num_other_actions*balance)

        return a0_downsampled + a1 + a2 + a3 + a4

    # Given a transition, this method converts its immediate reward into the dicounted return (discounted sum of
    # rewards) which can then be directly used as a target for the learning algorithm.
    def monteCarloTarget(self, transitions):
        reverseTransit = transitions[::-1]

        next_rew = 0

        for rt in reverseTransit:
            reward = rt["reward"]
            expected_dicounted_rewards = reward + GAMMA*next_rew

            rt["reward"] = expected_dicounted_rewards
            next_rew = expected_dicounted_rewards


# For creating and managing the dataset independently.
if __name__ == "__main__":
    #td = TransitionsDataset("./dataset_18.npy")

    # When we create the dataset from a folder of transition files we might encounter some issues due to "corrupted"
    # transitions where multiple actions were taken together by the demonstrator. For sake of simplicity I do no handle
    # this case and simply delete manually those transition files that present this problem during the dataset creation.
    td = TransitionsDataset("./transitions/", balance=None, mc=True)
