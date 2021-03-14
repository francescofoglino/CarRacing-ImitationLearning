from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import os
from PIL import Image
import numpy as np
import random

FRAME_SKIP  = 4
FRAME_STACK = 3

# Image transformation to be applied to each single image in our dataset. Greyscale allows us to downscale the image
# dimensionality to only one channel while normalization helps the training process
TRSF = transforms.Compose([transforms.ToTensor(),
                           transforms.Grayscale(),
                           transforms.Normalize((0.5), (0.5))])

# Custom dataset for handling and processing the transitions to be used for imitation learning
class TransitionsDataset(Dataset):
    # @balance is the variable for controlling the number of "no action" samples in the dataset since they represent the
    # vast majority of all the actions taken. This re-balancing is supposed to improve learning performance.
    def __init__(self, datasetPath, balance=None):
        # If we input a folder path the class will read all the transition files inside it for creating the dataset
        if os.path.isdir(datasetPath):
            transitions_files = os.listdir(datasetPath)
            data_name = "dataset_" + str(len(transitions_files)) + ".npy"

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
        elif action == [1./FRAME_SKIP, 0., 0.]:
            return 1
        # turn left
        elif action == [-1./FRAME_SKIP, 0., 0.]:
            return 2
        # accelerate
        elif action == [0., 1./FRAME_SKIP, 0.]:
            return 3
        # break
        elif action == [0., 0., 0.8/FRAME_SKIP]:
            return 4
        else:
            raise

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


# For creating and managing the dataset independently
if __name__ == "__main__":
    td = TransitionsDataset("./dataset_18.npy")
    #td = TransitionsDataset("./transitions/", balance=1)

    td.preProcessing()
