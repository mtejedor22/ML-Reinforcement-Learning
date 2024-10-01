import gym
import matplotlib.pyplot as plt
import numpy as np
import time


def extract_policy(environment, value, gamma):
    nA = environment.action_space.n  # Number of action
    nS = environment.observation_space.n  # Number of states
    policy = np.zeros(nS)
    for s in range(nS):
        Q_table = np.zeros(nA)
        for a in range(nA):
            Q_table[a] = sum([best_policy * (r + gamma * value[s_]) for best_policy, s_, r, _ in environment.P[s][a]])
        policy[s] = np.argmax(Q_table)
    return policy


def compute_policy_v(environment, policy, gamma):
    nA = environment.action_space.n  # Number of action
    nS = environment.observation_space.n  # Number of states
    value = np.zeros(nS)
    eps = 1e-5
    while True:
        prev_v = np.copy(value)
        for s in range(nS):
            policy_a = policy[s]
            value[s] = sum([best_policy * (r + gamma * prev_v[s_]) for best_policy, s_, r, is_done in environment.P[s][policy_a]])
        if (np.sum((np.fabs(prev_v - value))) <= eps):
            break
    return value

def run_episode(environment, policy, gamma, render=True):
    observable = environment.reset()
    observable = observable[0]
    total_reward = 0
    index_step = 0
    while True:
        if render:
            environment.render()
        observable, reward, terminated, truncated, info= environment.step(int(policy[observable]))
        done = terminated or truncated
        total_reward += (gamma ** index_step * reward)
        index_step += 1
        if done:
            break
    return total_reward


def evaluate_policy(environment, policy, gamma, n=500):
    scores = [run_episode(environment, policy, gamma, False) for _ in range(n)]
    return np.mean(scores)



def policy_iteration(environment, gamma):
    nA = environment.action_space.n  # Number of action
    nS = environment.observation_space.n  # Number of states
    state=0
    policy = np.random.choice(nA, size=(nS))
    max_iters = 200000
    desc = environment.unwrapped.desc
    for i in range(max_iters):
        start_time=time.time()
        old_policy_value = compute_policy_v(environment, policy, gamma)
        new_policy = extract_policy(environment, old_policy_value, gamma)
        if (np.all(policy == new_policy)):
            state = i + 1
            break
        policy = new_policy
    return policy, state


def value_iteration(environment, gamma):
    nA = environment.action_space.n  # Number of action
    nS = environment.observation_space.n  # Number of states
    state=0
    value = np.zeros(nS) 
    max_iters = 200000
    eps = 1e-20
    for i in range(max_iters):
        prev_v = np.copy(value)
        for s in range(nS):
            Q_table = [sum([best_policy * (r + gamma * prev_v[s_]) for best_policy, s_, r, _ in environment.P[s][a]]) for a in range(nA)]
            value[s] = max(Q_table)
        if (np.sum(np.fabs(prev_v - value)) <= eps):
            state = i + 1
            break
    return value, state

environment = 'FrozenLake-v1'   #16 states in grid world mdp
environment = gym.make(environment)
environment = environment.unwrapped
desc = environment.unwrapped.desc

times=np.empty(10)
discount_factor=np.empty(10)
iterations=np.empty(10)
score=np.empty(10)

for i in range(0, 10):
    gamma = (i + 0.5) / 10
    start_time = time.time()
    best_policy, state = policy_iteration(environment, gamma)
    scores = evaluate_policy(environment, best_policy, gamma)
    end_time = time.time()
    discount_factor[i] = (i + 0.5) / 10
    score[i] = scores
    iterations[i] = state
    times[i] = end_time - start_time


plt.plot(discount_factor, times)
plt.title('Frozen Lake - Policy Iteration Analysis: Discount Factor')
plt.ylabel('Time (s)')
plt.xlabel('Discount Factor')
plt.grid()
plt.savefig('images/FL_PI_DF.png')
#plt.show()
plt.close()

plt.plot(discount_factor, score)
plt.title('Frozen Lake - Policy Iteration Analysis: Average Rewards')
plt.ylabel('Average Rewards')
plt.xlabel('Discount Factor')
plt.grid()
plt.savefig('images/FL_PI_AR.png')
#plt.show()
plt.close()

plt.plot(discount_factor, iterations)
plt.title('Frozen Lake - Policy Iteration Analysis: Iterations to Converge')
plt.xlabel('Discount Factor')
plt.ylabel('Number of Iterations')
plt.grid()
plt.savefig('images/FL_PI_IC.png')
#plt.show()
plt.close()

times=np.empty(10)
discount_factor=np.empty(10)
iterations=np.empty(10)
score=np.empty(10)
best_val = np.empty(10)
for i in range(0, 10):
    gamma = (i + 0.5) / 10
    start_time = time.time()
    best_value, state = value_iteration(environment, gamma)
    policy = extract_policy(environment, best_value, gamma)
    scores = evaluate_policy(environment, policy, gamma, n=1000)
    end_time = time.time()
    discount_factor[i] = (i + 0.5) / 10
    score[i] = (scores)
    iterations[i] = state
    times[i] = end_time - start_time


plt.plot(discount_factor, times)
plt.title('Frozen Lake - Value Iteration Analysis: Discount Factor')
plt.ylabel('Time (s)')
plt.xlabel('Discount Factor')
plt.grid()
plt.savefig('images/FL_VI_DF.png')
#plt.show()
plt.close()

plt.plot(discount_factor, score)
plt.title('Frozen Lake - Value Iteration Analysis: Average Rewards')
plt.ylabel('Average Rewards')
plt.xlabel('Discount Factor')
plt.grid()
plt.savefig('images/FL_VI_AR.png')
#plt.show()
plt.close()

plt.plot(discount_factor, iterations)
plt.title('Frozen Lake - Policy Iteration Analysis: Iterations to Converge')
plt.xlabel('Discount Factor')
plt.ylabel('Number of Iterations')
plt.grid()
plt.savefig('images/FL_VI_IC.png')
#plt.show()
plt.close()










