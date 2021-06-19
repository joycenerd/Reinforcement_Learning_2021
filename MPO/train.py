from options import args
from mpo import MPO
import mujoco_py

from dm_control import suite
import quanser_robots
import gym


if __name__ == "__main__":
    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))
    input("Press enter to continue: ")

    # load env
    domain=args.domain
    task=args.task
    gym.logger.set_level(40)
    env = gym.make(args.env)
    # env=dm_control2gym.make(domain_name=domain, task_name=task, seed=1)
    # env=suite.load(domain_name=domain,task_name=task)

    # select model
    model = MPO(env)

    # model train
    model.train()
