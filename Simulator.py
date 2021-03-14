import gym
from gym import wrappers
from gym.wrappers import FrameStack
import random
import numpy as np
from DeepQNetwork import DeepQNetwork
from torchvision import transforms
import torch
import os
import Dataset

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

##########################################################################################
##########################################################################################
##########################################################################################

env_name = "CarRacing-v0"
env = gym.make(env_name)

FRAME_SKIP  = 4
FRAME_STACK = 3

env = SkipFrame(env, skip=FRAME_SKIP)
env = FrameStack(env, num_stack=FRAME_STACK)

#env = wrappers.Monitor(env, "./", video_callable=False ,force=True)

# This simple function applies the opposit transformation of convertActionBack
def convertActionBack(action):
    action = action.tolist()

    # no action
    if action == 0:
        return [0., 0., 0.]
    # turn right
    elif action == 1:
        return [1., 0., 0.]
    # turn left
    elif action == 2:
        return [-1., 0., 0.]
    # accelerate
    elif action == 3:
        return [0., 1., 0.]
    # break
    elif action == 4:
        return [0., 0., 0.8]
    else:
        raise

def runTestEpisodes(agent, num_episodes):

    env.render()
    #env.viewer.window.on_key_press = key_press
    #env.viewer.window.on_key_release = key_release

    #trsf = transforms.Compose([transforms.ToTensor(),
    #                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    returns = []

    for episode in range(num_episodes):
        # s_0 is the current state
        s_0 = env.reset()

        total_reward    = 0.0
        steps           = 0
        restart         = False

        while True:

            #input_img = TRSF(s_0.copy())
            input_img = Dataset.TransitionsDataset.transformState(s_0)

            a = agent(input_img[None, ...]).argmax()
            a = convertActionBack(a)

            #print(a)

            # s_1 is the next state also knowsn as s_prime
            s_1, r, done, info = env.step(a)
            total_reward += r

            steps += 1
            env.render()
            if done or restart:
                returns.append(total_reward)
                break

            # the current state becomes the next state
            s_0 = s_1

        episode += 1

    env.close()

    return returns

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
    agent.load_state_dict(torch.load("./models/sf0.5_lr1e-05_ut1/rep_0/DQN_e399.pt", map_location=torch.device('cpu')))

    R = runTestEpisodes(agent, 3)

    #computeAvgReturns("models/")
