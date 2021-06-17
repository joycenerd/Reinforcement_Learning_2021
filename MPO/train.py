from options import args
from mpo import MPO
import mujoco_py

import gym
import quanser_robots

if __name__ == "__main__":
    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))
    input("Press enter to continue: ")

    # load env
    gym.logger.set_level(40)
    env = gym.make(args.env)

    # select model
    model = MPO(env)

    # model train
    model.train()
