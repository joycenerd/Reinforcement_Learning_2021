# Policy Gradient Algorithms With Function Approximation

Implemenation of REINFORCE to solve CartPole-v0. Implementation of Advantage Actor-Critic algorithm to solve LunarLander-v2.

## Set up the environment

You should have Anaconda or Miniconda installed and Python>=3.7.

```
conda env create -f environment.yml
conda activate reinforce
```

## Reproducing the experiments

You should first create a directory name **preTrained**

```
python reinforce_baseline.py
python a2c.py
```