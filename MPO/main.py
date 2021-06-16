from options import args
from mpo import MPO

import gym


if __name__ == "__main__":
    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))
    input("Press enter to continue: ")

    # load env
    gym.logger.set_level(40)
    gym.make(args.env)
    model = MPO()
