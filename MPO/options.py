import argparse


parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Implementation of MPO on gym environments')
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--env', type=str, default='LunarLanderContinuous-v2',
                    help='gym environment')
parser.add_argument('--dual_constraint', type=float, default=0.1,
                    help='hard constraint of the dual formulation in the E-step')
parser.add_argument('--kl_mean_constraint', type=float, default=0.01,
                    help='hard constraint of the mean in the M-step')
parser.add_argument('--kl_var_constraint', type=float, default=0.0001,
                    help='hard constraint of the covariance in the M-step')
parser.add_argument('--kl_constraint', type=float, default=0.01,
                    help='hard constraint in the M-step')
parser.add_argument('--discount_factor', type=float, default=0.99,
                    help='discount factor used in Policy Evaluation')
parser.add_argument('--alpha_mean_scale', type=float, default=1.0,
                    help='scaling factor of the lagrangian multiplier in the M-step')
parser.add_argument('--alpha_var_scale', type=float, default=100.0,
                    help='scaling factor of the lagrangian multiplier in the M-step')
parser.add_argument('--alpha_scale', type=float, default=10.0,
                    help='scaling factor of the lagrangian multiplier in the M-step')
parser.add_argument('--alpha_mean_max', type=float, default=0.1,
                    help='maximum value of the lagrangian multiplier in the M-step')
parser.add_argument('--alpha_var_max', type=float, default=10.0,
                    help='maximum value of the lagrangian multiplier in the M-step')
parser.add_argument('--alpha_max', type=float, default=1.0,
                    help='maximum value of the lagrangian multiplier in the M-step')
parser.add_argument('--sample_episode_num', type=int, default=50,
                    help='number of episodes to learn')
parser.add_argument('--sample_episode_maxstep', type=int, default=600,
                    help='maximum sample steps of an episode')
parser.add_argument('--sample_action_num', type=int, default=64,
                    help='number of sampled actions')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--iteration_num', type=int, default=1000,
                    help='number of iteration to learn')
parser.add_argument('--episode_rerun_num', type=int, default=3,
                    help='number of reruns of sampled episode')
parser.add_argument('--mstep_iteration_num', type=int, default=5,
                    help='the number of iterations of the M-Step')
parser.add_argument('--evaluate_period', type=int, default=10,
                    help='periode of evaluation')
parser.add_argument('--evaluate_episode_num', type=int, default=100,
                    help='number of episodes to evaluate')
parser.add_argument('--evaluate_episode_maxstep', type=int, default=300,
                    help='maximum evaluate steps of an episode')
parser.add_argument('--log_dir', type=str, default="reacher_hard",
                    help='log directory')
parser.add_argument('--render', action='store_true')
parser.add_argument('--load', type=str, default="checkpoints/lunarlander_1000ep_0.4262rewards.pt",
                    help='load path')
parser.add_argument("--domain",type=str,default="reacher",help="domain name for control suite")
parser.add_argument("--task",type=str,default="hard",help="task name for control suite")
args = parser.parse_args()
