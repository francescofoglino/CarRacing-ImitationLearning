import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

import numpy as np
from math import floor

from DeepQNetwork import DeepQNetwork
from Dataset import TransitionsDataset, GAMMA
from Simulator import runTestEpisodes

import itertools
import random
import os

if __name__ == "__main__":

    print("TRAINING MONTE CARLO SCRIPT")

    NUMBER_PARAM_COMBS  = 1
    NUMBER_REPS         = 1

    NN_SCALE_PARAMETER  = [0.5]#[0.25, 0.5, 1]
    LEARNING_RATE       = [0.003]#[0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001]

    ALL_PARAM_COMBS     = list(itertools.product(*[NN_SCALE_PARAMETER, LEARNING_RATE]))
    param_combs         = random.sample(ALL_PARAM_COMBS, NUMBER_PARAM_COMBS)

    SAVE_MODEL          = 5

    # There are some small changes that need to be applied when running the experiments on colab
    colab = False

    root = ""

    if colab:
        root = "drive/MyDrive/Phantasma/"
    else:
        root = "./"

    # Dataset creation/loading
    transitions_dataset = TransitionsDataset(root + "dataset_mc_55.npy")
    transitions_dataset.preProcessing()

    losses_map = []

    for comb in param_combs:
        scale_factor, lr = comb

        master_folder_name = "models/sf" + str(scale_factor) + "_lr" + str(lr) + "/"

        print("Training models in folder : " + master_folder_name)

        for rep in range(NUMBER_REPS):
            folder_name = master_folder_name + "rep_" + str(rep) + "/"

            # Parameters
            num_epochs      = 200
            gamma           = GAMMA
            learning_rate   = lr
            batch           = 1000
            nn_scale        = scale_factor

            save_folder = root + folder_name
            os.makedirs(save_folder)

            transitions_dataset_loader  = DataLoader(transitions_dataset, batch_size=batch, shuffle=True)

            # I used the GPU only on Colab as my hardware was not powerful enough
            if colab:
                torch.set_default_tensor_type(torch.cuda.FloatTensor)

            # Instantiate a new Neural Network, Loss function and Optimizer
            net         = DeepQNetwork(nn_scale)

            criterion   = nn.MSELoss()
            optimizer   = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.)

            # For printing purposes
            epochs_loss = []

            num_batches   = len(transitions_dataset_loader)

            # Training loop
            for epoch in range(num_epochs):
                print("\n\nEpoch " + str(epoch))

                running_loss = 0.0

                for i, data in enumerate(transitions_dataset_loader):

                    # get the inputs and correspondent target values
                    s_0, a, s_1, discounted_R, final = data.values()

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    Q_0 = []

                    if colab:
                        Q_0 = net(s_0.to(torch.device('cuda')))
                    else:
                        Q_0 = net(s_0)

                    Q_0 = Q_0.gather(1,a.view(-1,1))

                    loss = criterion(Q_0, discounted_R.float().view(-1, 1))
                    loss.backward()

                    optimizer.step()

                    running_loss += loss.item()

                running_loss = running_loss / num_batches

                print("\nTraining Loss: " + str(running_loss))
                epochs_loss.append(running_loss)

                if (epoch+1)%SAVE_MODEL==0:
                    num_test_episodes = 1
                    R = runTestEpisodes(net, num_test_episodes)
                    print("Average Return : " + str(sum(R)/num_test_episodes))
                    # Save Neural Network Model
                    torch.save(net.state_dict(), save_folder + "DQN_e" + str(epoch) + ".pt")

            # Print and save the list of loss per epoch
            print(epochs_loss)
            np.savetxt(save_folder + "epochs.txt", epochs_loss)

        losses_map.append({save_folder : sum(epochs_loss)/num_epochs})

    print(list(losses_map))