import gym
import random
import numpy as np

from gym.wrappers import FrameStack

from Simulator import FRAME_STACK, FRAME_SKIP, SkipFrame

env_name = "CarRacing-v0"
env = gym.make(env_name)

env = SkipFrame(env, skip=FRAME_SKIP)
env = FrameStack(env, num_stack=FRAME_STACK)

if __name__ == "__main__":
    from pyglet.window import key

    a = np.array([0.0, 0.0, 0.0])

    def key_press(k, mod):
        global restart
        if k == 0xFF0D:
            restart = True
        if k == key.LEFT:
            a[0] = -1.0
        if k == key.RIGHT:
            a[0] = +1.0
        if k == key.UP:
            a[1] = +1.0
        if k == key.DOWN:
            a[2] = +0.8  # set 1.0 for wheels to block to zero rotation

    def key_release(k, mod):
        if k == key.LEFT and a[0] == -1.0:
            a[0] = 0
        if k == key.RIGHT and a[0] == +1.0:
            a[0] = 0
        if k == key.UP:
            a[1] = 0
        if k == key.DOWN:
            a[2] = 0

    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release

    transitions     = []
    episode         = 0

    isopen = True
    while isopen:
        # s_0 is the current state
        s_0 = env.reset()

        total_reward    = 0.0
        steps           = 0
        restart         = False

        while True:

            # s_1 is the next state also knowsn as s_prime
            #print(a)
            s_1, r, done, info = env.step(a)
            total_reward += r

            #print({"state_0": s_0, "action": a, "state_1": s_1, "reward": r, "final": done})
            transitions.append({"state_0": s_0, "action": a.copy(), "state_1": s_1, "reward": r, "final": done})

            #print(r)

            """
            if steps % 200 == 0 or done:
                print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            """

            steps += 1
            isopen = env.render()
            if done or restart or isopen == False:
                #print("\n\ndone : " + str(done) + "\nrestart : " + str(restart) + "\nisopen : " + str(isopen))
                break

            # the current state becomes the next state
            s_0 = s_1

        np.save("transitions" + str(episode) + ".npy", transitions)
        transitions = []

        episode += 1

    env.close()
