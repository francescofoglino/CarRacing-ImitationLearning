import gym
from gym import wrappers
from gym.wrappers import FrameStack
import random
import numpy as np
from DeepQNetwork import DeepQNetwork
from torchvision import transforms
import torch
import os
from Dataset import FRAME_STACK, FRAME_SKIP, TransitionsDataset, TURN, ACCELERATE, BREAK

#################################################################################################
# Function copied from link (https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html) #
#################################################################################################

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

#################################################################################################
#################################################################################################
#################################################################################################

# Cerate OpenAI gym environment
env_name = "CarRacing-v0"
env = gym.make(env_name)

# Downsample frame rate of the environment and stack multiple frames for creating a single observation
env = SkipFrame(env, skip=FRAME_SKIP)
env = FrameStack(env, num_stack=FRAME_STACK)

# This simple function applies the opposit transformation of convertAction in Dataset.py
def convertActionBack(action):
    action = action.tolist()

    # no action
    if action == 0:
        return [0., 0., 0.]
    # turn right
    elif action == 1:
        return [TURN/FRAME_SKIP, 0., 0.]
    # turn left
    elif action == 2:
        return [-TURN/FRAME_SKIP, 0., 0.]
    # accelerate
    elif action == 3:
        return [0., ACCELERATE/FRAME_SKIP, 0.]
    # break
    elif action == 4:
        return [0., 0., BREAK/FRAME_SKIP]
    else:
        raise

# Function for testing a single model @num_episodes times
def runTestEpisodes(agent, num_episodes):

    agent.eval()

    env.render()

    returns = []

    for episode in range(num_episodes):
        # s_0 is the current state
        s_0 = env.reset()

        total_reward    = 0.0
        restart         = False

        while True:

            input_img = TransitionsDataset.transformState(s_0)

            a = agent(input_img[None, ...]).argmax()
            a = convertActionBack(a)

            # s_1 is the next state also knowsn as s_prime
            s_1, r, done, info = env.step(a)
            total_reward += r

            env.render()
            if done or restart:
                returns.append(total_reward)
                break

            # the current state becomes the next state
            s_0 = s_1

        episode += 1

    env.close()

    agent.train()

    return returns

# This function runs a test for each model present in the input folder. It is ment to be used for testing each model
# saved during training of a single architecture in order to plot the correspondent learning curve
def computeAvgReturns(modelsPath, numTestEpisodes=3):
    import gc

    all_models = os.listdir(modelsPath)
    all_Rs = []

    agent = DeepQNetwork()

    for m in all_models:
        agent.load_state_dict(torch.load(modelsPath + m, map_location=torch.device('cpu')))

        R = runTestEpisodes(agent, numTestEpisodes)
        all_Rs.append(R)
        print(R)

        gc.collect()

    print(all_Rs)
    np.save(modelsPath + "Returns", all_Rs)

if __name__ == "__main__":
    agent = DeepQNetwork(0.5)
    #agent.load_state_dict(torch.load("./models/sf0.5_lr1e-05_ut1/rep_0/DQN_e399.pt", map_location=torch.device('cpu')))
    agent.load_state_dict(torch.load("./models/sf0.5_lr0.01_BN/DQN_e199.pt", map_location=torch.device('cpu')))

    R = runTestEpisodes(agent, 3)

    #computeAvgReturns("models/")
