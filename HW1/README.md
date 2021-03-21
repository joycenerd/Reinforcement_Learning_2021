# Policy Iteration and Value Iteration for MDP

## Introduction

Implementing policy iteration and value iteration for a classic MDP environment called "Taxi" (Dietterich 2000). This environment has been included in the OpenAI Gym: https://gym.openai.com/envs/Taxi-v3/. Implement the two functions `policy_iteration` and `value_iteration`. **Note that discrepancy=0 is a necessary condition of correct implementation, and with the default <img src="https://render.githubusercontent.com/render/math?math=\epsilon=10^{-3}">, you shall be able to observe zero discrepancy between the policies obtained by PI and VI**

## Highlights

* **Straight forward**: Easy implementation of VI and PI
* **Beginner Friendly**: It is beginner friendly to RL beginners

## Installation

You should have Anaconda or Miniconda installed in your environment

* `conda env create -f environment.yml`
* `conda activate Taxi`

Then you can run the code `policy_and_value_iteration.py` by `python policy_and_value_iteration.py`
