from torch.distributions import MultivariateNormal
import torch.nn.functional as F
import torch.nn as nn
import torch


class Actor(nn.Module):
    """
    Policy network
    :param env: (gym Environment) environment actor is operating on
    :param layer1: (int) size of the first hidden layer (default = 100)
    :param layer2: (int) size of the first hidden layer (default = 100)
    """

    def __init__(self, env, layer1, layer2):
        super(Actor, self).__init__()
        self.state_shape = env.observation_space.shape[0]
        if not env.action_space.shape:
            self.action_shape = env.action_space.n
        else:
            self.action_shape = env.action_space.shape[0]
        self.action_range = torch.from_numpy(env.action_space.high)

        self.linear1 = nn.Linear(self.state_shape, layer1, True)
        self.linear2 = nn.Linear(layer1, layer2, True)
        self.mean_layer = nn.Linear(layer2, self.action_shape, True)
        self.cholesky_layer = nn.Linear(layer2, self.action_shape, True)
        self.relu = nn.ReLU()

        self.cholesky = torch.zeros(self.action_shape, self.action_shape)

    def forward(self, states):
        """
        :param states: ([State]) a (batch of) state(s) of the environment
        :return: ([float])([float]) mean and cholesky factorization chosen by policy at given state
        """
        x = self.linear1(states)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)

        mean = self.action_range * torch.tanh(self.mean_layer(x))
        cholesky_vector = F.softplus(self.cholesky_layer(x))
        cholesky = torch.stack(
            [sigma * torch.eye(self.action_shape) for sigma in cholesky_vector])

        return mean, cholesky

    def get_action(self, state):
        """
        approximates an action by going forward through the network
        :param state: (State) a state of the environment
        :return: (float) an action of the action space
        """
        with torch.no_grad():
            mean, cholesky = self.forward(state)
            action_distribution = MultivariateNormal(mean, scale_tril=cholesky)
            action = action_distribution.sample()
        return action
