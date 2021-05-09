# Spring 2021, IOC 5269 Reinforcement Learning
# HW2: REINFORCE with baseline and A2C

import gym
from itertools import count
from collections import namedtuple
import numpy as nppytho

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.optim.lr_scheduler as Scheduler
from torch.utils.tensorboard import SummaryWriter
import argparse
import adabound


parser = argparse.ArgumentParser()
parser.add_argument(
    "--total-epochs", type=int, default=1000, help="total epochs to train reinforce"
)
parser.add_argument("--lr", type=float, default=0.01,
                    help="initial learning rate")
parser.add_argument(
    "--gamma", type=float, default=0.99, help="discounted reward factor"
)
args = parser.parse_args()

# Define a useful tuple (optional)
SavedAction = namedtuple("SavedAction", ["log_prob", "value"])

writer = SummaryWriter("runs/a2c", comment="Lunar_0.001")


class Actor(nn.Module):
    """
        Actor network
    """

    def __init__(self):
        super(Actor, self).__init__()

        # Extract the dimensionality of state and action spaces
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = (
            env.action_space.n if self.discrete else env.action_space.shape[0]
        )
        self.hidden_size = 128

        ########## YOUR CODE HERE (5~10 lines) ##########

        # Initialize the action layer
        self.fc1 = nn.Linear(self.observation_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, self.action_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        """
             Implement the forward pass for the action
        """

        ########## YOUR CODE HERE (3~5 lines) ##########

        x = self.fc1(state)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        action_prob = self.softmax(x)

        ########## END OF YOUR CODE ##########

        return action_prob


class Critic(nn.Module):
    """
        Critic network
    """

    def __init__(self):
        super(Critic, self).__init__()

        # Extract the dimensionality of state and action spaces
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = (
            env.action_space.n if self.discrete else env.action_space.shape[0]
        )
        self.hidden_size = 128

        ########## YOUR CODE HERE (5~10 lines) ##########

        # Initialize the action layer
        self.fc1 = nn.Linear(self.observation_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, state):
        """
             Implement the forward pass for the action
        """

        ########## YOUR CODE HERE (3~5 lines) ##########

        x = self.fc1(state)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        state_value = self.fc3(x)

        ########## END OF YOUR CODE ##########

        return state_value


def train(env, lr=0.001):
    """
        Train the model using SGD (via backpropagation)
        TODO: In each episode, 
        1. run the policy till the end of the episode and keep the sampled trajectory
        2. update both the policy and the value network at the end of episode
    """

    # Instantiate the policy model and the optimizer
    actor_model = Actor()
    critic_model = Critic()
    actor_optimizer = optim.Adam(actor_model.parameters(), lr=lr)
    critic_optimizer = optim.Adam(params=critic_model.parameters(), lr=lr)
    # actor_scheduler = Scheduler.StepLR(
    #  actor_optimizer, step_size=250, gamma=0.9)
    # critic_scheduler = Scheduler.StepLR(
    #   critic_optimizer, step_size=250, gamma=0.9)

    # Learning rate scheduler (optional)
    # scheduler = Scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    # EWMA reward for tracking the learning progress
    ewma_reward = 0

    # run inifinitely many episodes
    for i_episode in range(1, args.total_epochs + 1):
        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0
        t = 0
        # Uncomment the following line to use learning rate scheduler
        # scheduler.step()

        # For each episode, only run 9999 steps so that we don't
        # infinite loop while learning

        ########## YOUR CODE HERE (10-15 lines) ##########

        for t in range(1, 10000):
            probs = actor_model(torch.from_numpy(state).float())
            dist = Categorical(probs)
            action = dist.sample()
            state_val = critic_model(torch.from_numpy(state).float())
            state, reward, done, _ = env.step(action.detach().data.numpy())
            next_state_val = critic_model(torch.from_numpy(state).float())
            advantage = reward + (1 - done) * args.gamma * \
                next_state_val - state_val
            ep_reward += reward

            # update critic
            critic_loss = advantage.pow(2).mean()
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
            # critic_scheduler.step()

            # update actor
            # print("\n\n\n")
            # print(type(dist))
            # print("\n\n\n")
            actor_loss = -dist.log_prob(action) * advantage.detach()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()
            # actor_scheduler.step()

            if done:
                break

        writer.add_scalar("rewards/epochs", ep_reward, i_episode)
        ########## END OF YOUR CODE ##########

        # update EWMA reward and log the results
        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
        print(
            "Episode {}\tlength: {}\treward: {}\t ewma reward: {}".format(
                i_episode, t, ep_reward, ewma_reward
            )
        )

        # check if we have "solved" the cart pole problem

        if i_episode % 500 == 0:
            torch.save(
                actor_model.state_dict(),
                f"./preTrained/LunarLander_{i_episode}epochs.pth",
            )
        if ewma_reward > env.spec.reward_threshold:
            torch.save(
                actor_model.state_dict(), "./preTrained/LunarLander_{}.pth".format(lr)
            )
            print(
                "Solved! Running reward is now {} and "
                "the last episode runs to {} time steps!".format(
                    ewma_reward, t)
            )
            break


def test(name, env, n_episodes=10):
    """
        Test the learned model (no change needed)
    """
    model = Actor()

    model.load_state_dict(torch.load("./preTrained/{}".format(name)))

    render = True

    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        running_reward = 0
        for t in range(10000):
            probs = model(torch.from_numpy(state).float())
            dist = Categorical(probs)
            action = dist.sample()
            state, reward, done, _ = env.step(action.detach().data.numpy())
            running_reward += reward
            if render:
                env.render()
            if done:
                break
        print("Episode {}\tReward: {}".format(i_episode, running_reward))
    env.close()


if __name__ == "__main__":
    # For reproducibility, fix the random seed
    random_seed = 20
    lr = args.lr
    env = gym.make("LunarLander-v2")
    env.seed(random_seed)
    torch.manual_seed(random_seed)
    train(env, lr)
    writer.flush()
    test(f"LunarLander_{lr}.pth", env)
