from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import os
from PIL import Image
import numpy as np
import random
from Simulator import FRAME_SKIP

"""
TRSF = transforms.Compose([transforms.ToPILImage(),
                           transforms.Grayscale(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           transforms.ToTensor()])
"""

TRSF = transforms.Compose([transforms.ToTensor(),
                           transforms.Grayscale(),
                           transforms.Normalize((0.5), (0.5))])

class TransitionsDataset(Dataset):

    def __init__(self, datasetPath, downsample=False):

        if os.path.isdir(datasetPath):
            transitions_files = os.listdir(datasetPath)
            data_name = "dataset_Q_test_" + str(len(transitions_files)) + ".npy"

            self.dataset = []

            for t_file in transitions_files:
                transit = np.load(datasetPath + t_file, allow_pickle=True).tolist()

                if downsample:
                    transit     = self.reduceNumberFrames(transit)
                    data_name   = "dataset_" + str(len(transitions_files)) + "_d" + str(1000/len(transit)) + ".npy"

                for t in transit:

                    # Action conversion
                    a = t["action"]
                    t["action"] = self.convertAction(a)

                    self.dataset.append(t)

            self.dataset = self.balanceActionsSamples(self.dataset)

            np.save(data_name, self.dataset)

        elif os.path.isfile(datasetPath):
            self.dataset = np.load(datasetPath, allow_pickle=True)
        else:
            raise ValueError

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def preProcessing(self):
        for d in self.dataset:
            s_0 = d["state_0"]
            s_1 = d["state_1"]

            d["state_0"] = self.transformState(s_0)
            d["state_1"] = self.transformState(s_1)

            #self.transformTransition(d)

    @staticmethod
    def transformState(s):
        all_s = []

        num_frames = len(s)

        for f in range(num_frames):
            all_s.append(TRSF(s[f].copy()))

        return torch.cat(all_s)

    #obsolete
    def transformTransition(self, t):

        s_0 = t["state_0"]
        s_1 = t["state_1"]

        all_s_0 = []
        all_s_1 = []

        num_frames = len(s_0)

        for f in range(num_frames):
            all_s_0.append(TRSF(s_0[f]))
            all_s_1.append(TRSF(s_1[f]))

        t["state_0"] = torch.cat(all_s_0)
        t["state_1"] = torch.cat(all_s_1)

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

    # Obsolete. Simple function to perform downsampling on the dataset
    def reduceNumberFrames(self, transitions):

        new_transitions = []

        num_frames  = 2
        num_t       = len(transitions)/num_frames

        for i in range(int(num_t)):
            j = i*num_frames

            r_sum = 0
            for z in range(num_frames):
                r_sum += transitions[j+z]["reward"]

            down_transit = transitions[j+num_frames-1]
            down_transit["reward"] = r_sum

            new_transitions.append(down_transit)

        return new_transitions

    # This function balances the number of samples per action. Basically the vast majority of the actions is "no action"
    # due to the nature of the problem which can obstaculate learning. The if statement is "hard coded" on the number of
    # actions as they are not supposed to change in number and meaning.
    def balanceActionsSamples(self, transitions):

        balance = 1

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


# For creating and managing the dataset
if __name__ == "__main__":
    #td = TransitionsDataset("./dataset_D_25.npy")
    td = TransitionsDataset("./transitions/", downsample=False)

    td.preProcessing()

    sample = td[0]

    print("Hello")
    """
    This is an old test to visualize images before the transfromation
    
    for i in range(10):
        s_0, a, s_1, r = td[200+i].values()

        Image.fromarray(s_0, 'RGB').show()
        Image.fromarray(s_1, 'RGB').show()

        print(a)
        print(r)
    """
