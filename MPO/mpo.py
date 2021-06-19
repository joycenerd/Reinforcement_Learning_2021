from critic import Critic
from options import args
from actor import Actor

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import MultivariateNormal
from torch.utils.tensorboard import SummaryWriter
from scipy.optimize import minimize
import torch.nn as nn
import numpy as np
import torch


class MPO(object):
    def __init__(self, env):
        self.env = env
        self.eps = args.eps
        self.eps_mu = args.eps_mu
        self.eps_sigma = args.eps_sigma
        self.lr = args.lr
        self.alpha = args.alpha
        self.epochs = args.epochs
        self.steps = args.steps
        self.lagrange_iter = args.lagrange_iter
        self.batch_size = args.batch_size
        self.rerun_mb = args.rerun_mb
        self.sample_epochs = args.sample_epochs
        self.add_act = args.add_act
        self.actor_layers = args.policy_layers
        self.critic_layers = args.Q_layers
        self.log = args.log
        self.log_dir = args.log_dir
        self.render = args.render
        self.start_ep = 1
        self.checkpoint_dir = args.checkpoint_dir

        # initialize policy network
        # spec=self.env.action_spec()
        # self.action_range=torch.from_numpy(spec.maximum)
        self.action_shape = self.env.action_space.shape[0]
        self.action_range = torch.from_numpy(self.env.action_space.high)

        self.actor = Actor(self.env, self.actor_layers[0],
                           self.actor_layers[1])
        self.target_actor = Actor(self.env, self.actor_layers[0],
                                  self.actor_layers[1])

        for target_param, param in zip(self.target_actor.parameters(),
                                       self.actor.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                self.lr)

        self.mse_loss = nn.MSELoss()

        # initialize Q network
        self.critic = Critic(env,
                             layer1=self.critic_layers[0],
                             layer2=self.critic_layers[1])
        self.target_critic = Critic(env,
                                    layer1=self.critic_layers[0],
                                    layer2=self.critic_layers[1])

        for target_param, param in zip(self.target_critic.parameters(),
                                       self.critic.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.lr)

        # initialize Lagrange Multiplier
        self.eta = np.random.rand()
        self.eta_mu = np.random.rand()
        self.eta_sigma = np.random.rand()

    def train(self):
        # initialize flags and params
        rend = self.render
        ep = self.epochs
        it = self.steps
        L = self.sample_epochs
        rerun = self.rerun_mb

        # initialize logging
        is_log = self.log
        log_d = self.log_dir
        if is_log:
            writer = (SummaryWriter()
                      if log_d is None else SummaryWriter("runs/" + log_d))

        # start training
        for epoch in range(self.start_ep, ep + 1):
            # update replay buffer
            (
                states,
                actions,
                rewards,
                next_states,
                mean_reward,
            ) = self._sample_trajectory(L, it, rend)
            mean_q_loss = 0
            mean_lagrange = 0

            # Find better policy by gradient descent
            for _ in range(rerun):
                for indices in BatchSampler(SubsetRandomSampler(range(it)),
                                            self.batch_size, False):
                    state_batch = states[indices]
                    action_batch = actions[indices]
                    reward_batch = rewards[indices]
                    next_state_batch = next_states[indices]

                    # sample M additional action for each state
                    target_mu, target_A = self.target_actor.forward(
                        torch.tensor(state_batch).float())
                    target_mu.detach()
                    target_A.detach()
                    action_distribution = MultivariateNormal(
                        target_mu, scale_tril=target_A)
                    additional_action = []
                    additional_target_q = []
                    additional_next_q = []
                    additional_q = []

                    for i in range(self.add_act):
                        action = action_distribution.sample()
                        additional_action.append(action)
                        additional_target_q.append(
                            self.target_critic.forward(
                                torch.tensor(state_batch).float(),
                                action).detach().numpy())
                        additional_next_q.append(
                            self.target_critic.forward(
                                torch.tensor(next_state_batch).float(),
                                action).detach())
                        additional_q.append(
                            self.critic.forward(
                                torch.tensor(state_batch).float(), action))

                    additional_action = torch.stack(
                        additional_action).squeeze()
                    additional_q = torch.stack(additional_q).squeeze()
                    additional_target_q = np.array(
                        additional_target_q).squeeze()
                    additional_next_q = torch.stack(
                        additional_next_q).squeeze()

                    mean_q = torch.mean(additional_q, 0)
                    mean_next_q = torch.mean(additional_next_q, 0)

                    # Update Q-function
                    q_loss = self._critic_update(
                        states=state_batch,
                        rewards=reward_batch,
                        actions=action_batch,
                        mean_next_q=mean_next_q,
                    )
                    mean_q_loss += q_loss

                    # E-step
                    # Update Dual-function
                    def dual(eta):
                        """
                        Dual function of the non-parametric variational
                        g(η) = η*ε + η \sum \log (\sum \exp(Q(a, s)/η))
                        """
                        max_q = np.max(additional_target_q, 0)
                        return (eta * self.eps + np.mean(max_q) +
                                eta * np.mean(
                                    np.log(
                                        np.mean(
                                            np.exp(
                                                (additional_target_q - max_q) /
                                                eta), 0))))

                    bounds = [(1e-6, 1e6)]
                    res = minimize(dual,
                                   np.array([self.eta]),
                                   method="SLSQP",
                                   bounds=bounds)
                    self.eta = res.x[0]

                    # calculate the new q values
                    exp_Q = torch.tensor(additional_target_q) / self.eta
                    baseline = torch.max(exp_Q, 0)[0]
                    exp_Q = torch.exp(exp_Q - baseline)
                    normalization = torch.mean(exp_Q, 0)
                    action_q = additional_action * exp_Q / normalization
                    action_q = np.clip(action_q,
                                       a_min=-self.action_range,
                                       a_max=self.action_range)

                    # M-step
                    # update policy based on lagrangian
                    for _ in range(self.lagrange_iter):
                        mu, A = self.actor.forward(
                            torch.tensor(state_batch).float())
                        policy = MultivariateNormal(mu, scale_tril=A)

                        additional_logprob = []
                        if self.add_act == 1:
                            additional_logprob = policy.log_prob(action_q)
                        else:
                            for column in range(self.add_act):
                                action_vec = action_q[column, :]
                                action_vec = action_vec.unsqueeze(1)
                                additional_logprob.append(
                                    policy.log_prob(action_vec))
                            additional_logprob = torch.stack(
                                additional_logprob).squeeze()
                        C_mu, C_sigma = self._calculate_gaussian_kl(
                            actor_mean=mu,
                            target_mean=target_mu,
                            actor_cholesky=A,
                            target_cholesky=target_A,
                        )

                        # Update lagrange multipliers by gradient descent
                        self.eta_mu -= self.alpha * (self.eta_mu -
                                                     C_mu).detach().item()
                        self.eta_sigma -= (
                            self.alpha *
                            (self.eta_sigma - C_sigma).detach().item())
                        if self.eta_mu < 0:
                            self.eta_mu = 0
                        if self.eta_sigma < 0:
                            self.eta_sigma = 0

                        self.actor_optimizer.zero_grad()
                        loss_policy = -(torch.mean(additional_logprob) +
                                        self.eta_mu *
                                        (self.eta_mu - C_mu) + self.eta_sigma *
                                        (self.eta_sigma - C_sigma))
                        mean_lagrange += loss_policy.item()
                        loss_policy.backward()
                        self.actor_optimizer.step()

            self._update_param()

            print(
                "\n Epoch:\t",
                epoch,
                "\n Mean reward:\t",
                mean_reward / it / L,
                "\n Mean Q loss:\t",
                mean_q_loss / 50,
                "\n Mean Lagrange:\t",
                mean_lagrange / 50,
                "\n η:\t",
                self.eta,
                "\n η_μ:\t",
                self.eta_mu,
                "\n η_Σ:\t",
                self.eta_sigma,
            )

            # saving and logging
            if is_log:
                # number_mb = int(self.it / self.mb_size) + 1
                reward_target = self.eval(10, it, render=False)
                writer.add_scalar('target/mean_rew_10_ep', reward_target,
                                  epoch)
                writer.add_scalar('data/mean_reward', mean_reward, epoch)
                writer.add_scalar('data/mean_lagrangeloss',
                                  mean_lagrange / self.lagrange_iter / 50,
                                  epoch)
                writer.add_scalar('data/mean_qloss', mean_q_loss / 50, epoch)

            if epoch % 5 == 0:
                data = {
                    'epoch': epoch,
                    'critic_state_dict': self.critic.state_dict(),
                    'target_critic_state_dict':
                    self.target_critic.state_dict(),
                    'actor_state_dict': self.actor.state_dict(),
                    'target_actor_state_dict': self.target_actor.state_dict(),
                    'critic_optim_state_dict':
                    self.critic_optimizer.state_dict(),
                    'actor_optim_state_dict':
                    self.actor_optimizer.state_dict()
                }
                save_path = self.checkpoint_dir + f"model_{epoch}epoch_{reward_target}rewards.pth"
                torch.save(data, save_path)

        # end training
        if is_log:
            writer.close()

    def eval(self, epochs, steps, render=True):
        """
        method for evaluating current model (mean reward for a given number of
        episodes and episode length)
        :param episodes: (int) number of episodes for the evaluation
        :param episode_length: (int) length of a single episode
        :param render: (bool) flag if to render while evaluating
        :return: (float) meaned reward achieved in the episodes
        """

        summed_rewards = 0
        for episode in range(epochs):
            reward = 0
            observation = self.env.reset()
            for step in range(steps):
                action = self.target_actor.eval_step(observation)
                new_observation, rew, done, _ = self.env.step(
                    action.detach().numpy())
                reward += rew
                if render:
                    self.env.render()
                observation = new_observation if not done else self.env.reset()

            summed_rewards += reward
        return summed_rewards / epochs

    def _update_param(self):
        """
        Sets target parameters to trained parameter
        """
        # Update policy parameters
        for target_param, param in zip(self.target_actor.parameters(),
                                       self.actor.parameters()):
            target_param.data.copy_(param.data)

        # Update critic parameters
        for target_param, param in zip(self.target_critic.parameters(),
                                       self.critic.parameters()):
            target_param.data.copy_(param.data)

    def _calculate_gaussian_kl(self, actor_mean, target_mean, actor_cholesky,
                               target_cholesky):
        """
        calculates the KL between the old and new policy assuming a gaussian distribution
        :param actor_mean: ([float]) mean of the actor
        :param target_mean: ([float]) mean of the target actor
        :param actor_cholesky: ([[float]]) cholesky matrix of the actor covariance
        :param target_cholesky: ([[float]]) cholesky matrix of the target actor covariance
        :return: C_μ, C_Σ: ([float],[[float]])mean and covariance terms of the KL
        """
        inner_sigma = []
        inner_mu = []
        for mean, target_mean, a, target_a in zip(actor_mean, target_mean,
                                                  actor_cholesky,
                                                  target_cholesky):
            sigma = a @ a.t()
            target_sigma = target_a @ target_a.t()
            inverse = sigma.inverse()
            inner_sigma.append(
                torch.trace(inverse @ target_sigma) - sigma.size(0) +
                torch.log(sigma.det() / target_sigma.det()))
            inner_mu.append(
                (mean - target_mean) @ inverse @ (mean - target_mean))

        inner_mu = torch.stack(inner_mu)
        inner_sigma = torch.stack(inner_sigma)
        C_μ = 0.5 * torch.mean(inner_mu)
        C_Σ = 0.5 * torch.mean(inner_sigma)

        return C_μ, C_Σ

    def _critic_update(self, states, rewards, actions, mean_next_q):
        """
        Updates the critics
        :param states: ([State]) mini-batch of states
        :param actions: ([Action]) mini-batch of actions
        :param rewards: ([Reward]) mini-batch of rewards
        :param mean_next_q: ([State]) target Q values
        :return: (float) q-loss
        """
        rewards = torch.from_numpy(rewards).float()
        mean_next_q = mean_next_q.unsqueeze(1)
        y = rewards + self.lr * mean_next_q
        self.critic_optimizer.zero_grad()
        target = self.critic(
            torch.from_numpy(states).float(),
            torch.from_numpy(actions).float())
        prev = y[0]
        loss_critic = self.mse_loss(y, target)
        loss_critic.backward()
        self.critic_optimizer.step()
        return loss_critic.item()

    def _sample_trajectory(self, epochs, epoch_length, render):
        """
        Samples a trajectory which serves as a batch
        :param epochs: (int) number of epochs to be sampled
        :param epoch_length: (int) length of a single epoch
        :param render: (bool) flag if steps should be rendered
        :return: [States], [Action], [Reward], [State]: batch of states, actions, rewards and next-states
        """
        states = []
        rewards = []
        actions = []
        next_states = []
        mean_reward = 0

        for _ in range(epochs):
            observation = self.env.reset()
            for steps in range(epoch_length):
                action = self.target_actor.get_action(
                    torch.from_numpy(observation).float())
                action = np.reshape(action.numpy(), -1)
                new_observation, reward, done, _ = self.env.step(action)
                mean_reward += reward
                if render:
                    self.env.render()
                states.append(observation)
                rewards.append(reward)
                actions.append(action)
                next_states.append(new_observation)
                if done:
                    observation = self.env.reset()
                else:
                    observation = new_observation

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        return states, actions, rewards, next_states, mean_reward
