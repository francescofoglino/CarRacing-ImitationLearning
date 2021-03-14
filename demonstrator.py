import gym
import random
import numpy as np

from gym.wrappers import FrameStack
from Simulator import SkipFrame
from Dataset import FRAME_STACK, FRAME_SKIP

# Cerate OpenAI gym environment
env_name = "CarRacing-v0"
env = gym.make(env_name)

# Downsample frame rate of the environment and stack multiple frames for creating a single observation
env = SkipFrame(env, skip=FRAME_SKIP)
env = FrameStack(env, num_stack=FRAME_STACK)

if __name__ == "__main__":
    ##########################################################################################
    # Given user keyboard command controller for controlling the agent during demonstrations #
    ##########################################################################################
    from pyglet.window import key

    a = np.array([0.0, 0.0, 0.0])


    # The SkipFrame wrapper accumulates the actions of the frames it skips. In order to better control the car
    # I divide the value of the action by the number of the skipped frames so that it is "normalized" to the
    # original value in the code provided for controlling the car.
    def key_press(k, mod):
        global restart
        if k == 0xFF0D:
            restart = True
        if k == key.LEFT:
            a[0] = -1.0/FRAME_SKIP
        if k == key.RIGHT:
            a[0] = +1.0/FRAME_SKIP
        if k == key.UP:
            a[1] = +1.0/FRAME_SKIP
        if k == key.DOWN:
            a[2] = +0.8/FRAME_SKIP  # set 1.0 for wheels to block to zero rotation

    def key_release(k, mod):
        if k == key.LEFT and a[0] == -1.0/FRAME_SKIP:
            a[0] = 0
        if k == key.RIGHT and a[0] == +1.0/FRAME_SKIP:
            a[0] = 0
        if k == key.UP:
            a[1] = 0
        if k == key.DOWN:
            a[2] = 0

    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release

    ##########################################################################################
    ##########################################################################################
    ##########################################################################################

    # @episode is used as an ID for saving the different demonstrations in different files
    episode         = 15

    isopen = True

    while isopen:
        # s_0 is the current state
        s_0 = env.reset()

        total_reward    = 0.0
        restart         = False

        # @transitions is the list that gets filled up by each observed transition during the demonstration.
        transitions = []

        while True:

            # s_1 is the next state
            s_1, r, done, info = env.step(a)
            total_reward += r

            # Store current transition for saving it to file. It will be used for creating the dataset for training
            transitions.append({"state_0": s_0, "action": a.copy(), "state_1": s_1, "reward": r, "final": done})

            isopen = env.render()
            if done or restart or isopen == False:
                break

            # the current state becomes the next state
            s_0 = s_1

        np.save("transitions" + str(episode) + ".npy", transitions)

        episode += 1

    env.close()
