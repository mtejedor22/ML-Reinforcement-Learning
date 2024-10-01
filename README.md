# Reinforcement Learning and Markov Decision Process (MDP)

## Introduction
This repository contains code for implementing various reinforcement learning algorithms and solving Markov Decision Process (MDP) problems. Reinforcement learning is a type of machine learning where an agent learns to make decisions by interacting with an environment to achieve a certain goal. Markov Decision Process is a mathematical framework used to model decision-making in situations where outcomes are partly random and partly under the control of a decision-maker.

For accessing the files through Georgia Tech affiliation, please use the following link: [https://gatech.box.com/s/t7t7fmp93vgpi0bk69jkz9aco6ptv2kr](https://gatech.box.com/s/t7t7fmp93vgpi0bk69jkz9aco6ptv2kr)

## Dependencies
Before running the code, ensure you have the following libraries installed:

```bash
pip install numpy
```
```bash
pip install matplotlib
```
```bash
pip install gym
```
```bash
pip install mdptoolbox-hiive
```

## Overview

This project focuses on the implementation and comparison of three fundamental reinforcement learning algorithms:

1. **Policy Iteration:** A dynamic programming technique that iteratively evaluates and improves the policy until convergence, guaranteeing convergence to the optimal policy.
2. **Value Iteration:** Another dynamic programming method that iteratively computes the value function until convergence, ultimately yielding the optimal policy.
3. **Q-Learning:** A model-free reinforcement learning algorithm that learns the optimal policy through trial and error without requiring a model of the environment.

The performance of these algorithms is evaluated using a standard test environment provided by the OpenAI Gym toolkit.

## Usage

1. Open the provided code inside the folder in a compatible environment.
2. Ensure you have all necessary dependencies installed.
3. Execute the code files and observe the results and images generated.
