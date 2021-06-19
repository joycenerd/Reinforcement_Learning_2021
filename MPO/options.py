import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    type=str,
    default="Hopper-v2",
    help=
    "(str) which task in the OpenAI gym to do: [BipedalWalker-v2, Hopper-v2, Acrobot-v1, CartPole-v0,Levitation-v0]"
)
parser.add_argument(
    "--eps",
    type=float,
    default=0.1,
    help="(float) hard constraint of the dual formulation in the E-step")
parser.add_argument("--eps-mu",
                    type=float,
                    default=0.1,
                    help="(float) hard constraint for mean in the M-step")
parser.add_argument(
    "--eps-sigma",
    type=float,
    default=1e-4,
    help="(float) hard constraint for covarianve in the M-step")
parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
parser.add_argument(
    "--alpha",
    type=float,
    default=10,
    help="(float) scaling factor of the lagrangian multiplier in the M-step")
parser.add_argument("--epochs",
                    type=int,
                    default=1000,
                    help="(int) number of training epochs")
parser.add_argument("--steps",
                    type=int,
                    default=1000,
                    help="(int) number of steps in each epoch")
parser.add_argument("--lagrange-iter",
                    type=int,
                    default=5,
                    help="number of optimization steps of the Lagrangian")
parser.add_argument("--batch-size",
                    type=int,
                    default=64,
                    help="(int) size of the sampled mini-batch size")
parser.add_argument("--rerun-mb",
                    type=int,
                    default=5,
                    help="(int) number of reruns of the mini batch")
parser.add_argument("--sample-epochs",
                    type=int,
                    default=1,
                    help="(int) number of sampling episodes")
parser.add_argument("--add_act",
                    type=int,
                    default=64,
                    help="(int) number of additional actions")
parser.add_argument("--policy_layers",
                    type=int,
                    default=(100, 100),
                    help="(int,int) number of hidden layers in the policy net")
parser.add_argument(
    "--Q-layers",
    type=int,
    default=(200, 200),
    help="(int,int) number of hidden layers in the Q function net")
parser.add_argument("--log",
                    type=bool,
                    default=True,
                    help="(bool) saves log if True")
parser.add_argument("--log-dir",
                    type=str,
                    default="Hopper-v2",
                    help="(str) log save dir name")
parser.add_argument("--render",
                    type=bool,
                    default=False,
                    help="(bool) renders the simulation if True")
parser.add_argument("--checkpoint-dir",
                    type=str,
                    default="./checkpoints/",
                    help="folder to save checkpoints")
parser.add_argument("--domain",type=str,default="cartpole",help="control suite environment domain name")
parser.add_argument("--task",type=str,default="balance",help="control suite environment task name")
args = parser.parse_args()
