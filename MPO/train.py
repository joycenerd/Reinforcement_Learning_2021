from options import args
from mpo import MPO

import dm_control2gym
import mujoco_py
import gym


def main():
    gym.logger.set_level(40)
    env = gym.make(args.env)
    # env = dm_control2gym.make(domain_name=args.domain, task_name=args.task)

    model = MPO(
        args.device,
        env,
        log_dir=args.log_dir,
        dual_constraint=args.dual_constraint,
        kl_mean_constraint=args.kl_mean_constraint,
        kl_var_constraint=args.kl_var_constraint,
        kl_constraint=args.kl_constraint,
        discount_factor=args.discount_factor,
        alpha_mean_scale=args.alpha_mean_scale,
        alpha_var_scale=args.alpha_var_scale,
        alpha_scale=args.alpha_scale,
        alpha_mean_max=args.alpha_mean_max,
        alpha_var_max=args.alpha_var_max,
        alpha_max=args.alpha_max,
        sample_episode_num=args.sample_episode_num,
        sample_episode_maxstep=args.sample_episode_maxstep,
        sample_action_num=args.sample_action_num,
        batch_size=args.batch_size,
        episode_rerun_num=args.episode_rerun_num,
        mstep_iteration_num=args.mstep_iteration_num,
        evaluate_period=args.evaluate_period,
        evaluate_episode_num=args.evaluate_episode_num,
        evaluate_episode_maxstep=args.evaluate_episode_maxstep)

    if args.load is not None:
        model.load_model(args.load)
    

    model.train(
        iteration_num=args.iteration_num,
        log_dir=args.log_dir,
        render=args.render)
    
    # model.test()

    env.close()


if __name__ == '__main__':
    main()