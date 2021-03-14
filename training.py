import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

import numpy as np
from math import floor

from DeepQNetwork import DeepQNetwork
from Dataset import TransitionsDataset
from Simulator import runTestEpisodes

import itertools
import random
import os

if __name__ == "__main__":

    print("TRAINING SCRIPT")

    NUMBER_PARAM_COMBS  = 1
    NUMBER_REPS         = 1

    NN_SCALE_PARAMETER  = [1]#[0.25, 0.5, 1]
    LEARNING_RATE       = [0.0003]#[0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001]
    UPDATE_TARGET       = [50]#[5, 10, 20, 50]

    ALL_PARAM_COMBS     = list(itertools.product(*[NN_SCALE_PARAMETER, LEARNING_RATE, UPDATE_TARGET]))
    param_combs         = random.sample(ALL_PARAM_COMBS, NUMBER_PARAM_COMBS)

    SAVE_MODEL          = 10

    # There are some small changes that need to be applied when running the experiments on colab
    colab = False

    root = ""

    if colab:
        root = "drive/MyDrive/Phantasma/"
    else:
        root = "./"

    # Dataset creation/loading
    transitions_dataset = TransitionsDataset(root + "dataset_Q_test_11.npy")
    transitions_dataset.preProcessing()

    losses_map = []

    for comb in param_combs:
        scale_factor, lr, up_target = comb

        master_folder_name = "models/sf" + str(scale_factor) + "_lr" + str(lr) + "_ut" + str(up_target) + "/"

        print("Training models in folder : " + master_folder_name)

        for rep in range(NUMBER_REPS):
            folder_name = master_folder_name + "rep_" + str(rep) + "/"

            # Parameters
            num_epochs      = 500
            gamma           = 0.99
            learning_rate   = lr
            batch           = 1000
            update_target   = up_target
            nn_scale        = scale_factor

            save_folder = root + folder_name
            os.makedirs(save_folder)

            transitions_dataset_loader  = DataLoader(transitions_dataset, batch_size=batch, shuffle=True)

            # I used the GPU only on Colab as my hardware was not powerful enough
            if colab:
                torch.set_default_tensor_type(torch.cuda.FloatTensor)

            # Instantiate a new Neural Network, Loss function and Optimizer
            net         = DeepQNetwork(nn_scale)
            target_net  = DeepQNetwork(nn_scale)
            target_net.load_state_dict(net.state_dict())
            target_net.eval()

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
                    s_0, a, s_1, r, final = data.values()

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    Q_0 = []
                    Q_1 = []

                    if colab:
                        Q_0 = net(s_0.to(torch.device('cuda')))
                        Q_1 = target_net(s_1.to(torch.device('cuda'))).detach()
                    else:
                        Q_0 = net(s_0)
                        Q_1 = target_net(s_1).detach()

                    Q_0 = Q_0.gather(1,a.view(-1,1))
                    Q_1 = Q_1.max(1)[0].view(-1,1)
                    Q_1[final] = 0 #set terminal state to 0

                    target = (gamma*Q_1) + r.view(-1,1)

                    # targets need to be converted to float in order to perform the backward pass
                    #targets = targets.float().view(-1, 1)

                    loss = criterion(Q_0, target.float())
                    loss.backward()

                    optimizer.step()

                    running_loss += loss.item()

                running_loss = running_loss / num_batches

                print("\nTraining Loss: " + str(running_loss))
                epochs_loss.append(running_loss)

                if (epoch+1)%update_target==0:
                    target_net.load_state_dict(net.state_dict())

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
    np.savetxt(root + "all_losses.txt", list(losses_map), dtype=object)