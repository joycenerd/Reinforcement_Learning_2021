# Spring 2021, IOC 5269 Reinforcement Learning
# HW1-PartII: First-Visit Monte-Carlo and Temporal-difference policy evaluation

import gym
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict
import sys



env = gym.make("Blackjack-v0")

def mc_policy_evaluation(policy, env, num_episodes, gamma=1.0):
    """
        Find the value function for a given policy using first-visit Monte-Carlo sampling
        
        Input Arguments
        ----------
            policy: 
                a function that maps a state to action probabilities
            env:
                an OpenAI gym environment
            num_episodes: int
                the number of episodes to sample
            gamma: float
                the discount factor
        ----------
        
        Output
        ----------
            V: dict (that maps from state -> value)
        ----------
    
        TODOs
        ----------
            1. Initialize the value function
            2. Sample an episode and calculate sample returns
            3. Iterate and update the value function
        ----------
        
    """
    
    # value function -> initialize the value function
    V = defaultdict(float)
    
    ##### FINISH TODOS HERE #####

    # keep track of returns (sum and count)
    returns_sum = defaultdict(float)
    returns_cnt = defaultdict(float)

    for episode in range(1,num_episodes+1):
        if episode%1000==0:
            print(f'Episode: {episode}/{num_episodes}')
            print('-'*len(f'Episode: {episode}/{num_episodes}'))
            sys.stdout.flush()


        # sample an episode (trajectory) and calculate sample return
        trajectory=[] # array of tuples (state,action,reward)
        state=env.reset()
        for t in range(100):
            action=policy(state)
            next_state,reward,done,_=env.step(action)
            trajectory.append((state,action,reward))
            # print(f'state: {state} action: {action} reward: {reward}')
            if done:
                break
            state=next_state

        # find all states that we have visited in this episode (trajectory)
        state_in_trajectory=set()
        for state,action,reward in trajectory:
            state_in_trajectory.add(state)
        for state in state_in_trajectory:
            for i,x in enumerate(trajectory):
                if x[0]==state:
                    first_occurrence_idx=i # find the first occurrence of state in the trajectory
                    break
            # Iterate and update the value function
            G=0.0
            for i,x in enumerate(trajectory[first_occurrence_idx:]):
                G+=gamma**i*x[2]
            returns_sum[state]+=G
            returns_cnt[state]+=1.0
            V[state]=returns_sum[state]/returns_cnt[state]

    #############################

    return V


def td0_policy_evaluation(policy, env, num_episodes, gamma=1.0):
    """
        Find the value function for the given policy using TD(0)
    
        Input Arguments
        ----------
            policy: 
                a function that maps a state to action probabilities
            env:
                an OpenAI gym environment
            num_episodes: int
                the number of episodes to sample
            gamma: float
                the discount factor
        ----------
    
        Output
        ----------
            V: dict (that maps from state -> value)
        ----------
        
        TODOs
        ----------
            1. Initialize the value function
            2. Sample an episode and calculate TD errors
            3. Iterate and update the value function
        ----------
    """
    # value function -> initialize the value function
    V = defaultdict(float)

    ##### FINISH TODOS HERE #####
    ALPHA=1.0
    returns_sum=defaultdict(float)
    returns_cnt=defaultdict(float)
    for episode in range(1,num_episodes+1):
        if episode%1000==0:
            print(f'Episode: {episode}/{num_episodes}')
            print('-'*len(f'Episode: {episode}/{num_episodes}'))
            sys.stdout.flush()

        state=env.reset()
        while True:

            # Sample an episode and calculate TD errors
            action=policy(state)
            next_state,reward,done,_=env.step(action)
            td_target=reward+gamma*V[next_state]
            td_error=td_target-V[state]

            # Iterate and update the value function
            returns_cnt[state] += 1.0
            V[state]=(V[state]*(returns_cnt[state]-1)+ALPHA*td_error)/returns_cnt[state]
            if done:
                break
            state=next_state
    #############################

    return V


def plot_value_function(V, title="Value Function"):
    """
        Plots the value function as a surface plot.
        (Credit: Denny Britz)
    """
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    # Find value for all (x, y) coordinates
    Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))
    Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))

    def plot_surface(X, Y, Z, title):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player Sum')
        ax.set_ylabel('Dealer Showing')
        ax.set_zlabel('Value')
        ax.set_title(title)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)
        plt.show()

    plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title))
    plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title))
    
    
def apply_policy(observation):
    """
        A policy under which one will stick if the sum of cards is >= 20 and hit otherwise.
    """
    score, dealer_score, usable_ace = observation
    return 0 if score >= 20 else 1


if __name__ == '__main__':
    
    V_mc_10k = mc_policy_evaluation(apply_policy, env, num_episodes=10000)
    plot_value_function(V_mc_10k, title="10,000 Steps")
    V_mc_500k = mc_policy_evaluation(apply_policy, env, num_episodes=500000)
    plot_value_function(V_mc_500k, title="500,000 Steps")


    V_td0_10k = td0_policy_evaluation(apply_policy, env, num_episodes=10000)
    plot_value_function(V_td0_10k, title="10,000 Steps")
    V_td0_500k = td0_policy_evaluation(apply_policy, env, num_episodes=500000)
    plot_value_function(V_td0_500k, title="500,000 Steps")
    



