import torch.nn as nn
import torch


class Critic(nn.Module):
    """
    Critic class for Q-function network
    :param env: (gym Environment) environment network is operating on
    :param layer1: (int) size of the first hidden layer (default = 200)
    :param layer2: (int) size of the first hidden layer (default = 200)
    """

    def __init__(self, env, layer1=200, layer2=200):
        super(Critic, self).__init__()
        self.state_shape = env.observation_space.shape[0]
        self.action_shape = env.action_space.shape[0]

        self.linear1 = nn.Linear(self.state_shape, layer1, True)
        self.linear2 = nn.Linear(layer1 + self.action_shape, layer2, True)
        self.linear3 = nn.Linear(layer2, 1, True)
        self.relu = nn.ReLU()

    def forward(self, state, action):
        """
        :param state: (State) a state of the environment
        :param action: (Action) an action of the environments action-space
        :return: (float) Q-value for the given state-action pair
        """
        x = self.linear1(state)
        x = self.relu(x)
        x = torch.cat((x, action), 1)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)

        return x
