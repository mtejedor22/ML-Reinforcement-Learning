import gym
import matplotlib.pyplot as plt
import numpy as np
import time


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

def policy_iteration(environment, gamma):
    nA = environment.action_space.n  # Number of action
    nS = environment.observation_space.n  # Number of states
    state=0
    policy = np.random.choice(nA, size=(nS))
    max_iters = 200000
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
    value = np.zeros(nS)  # initialize value-function
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

reward_array = []
r_array=[]
iter_array = []
size_array = []
ts_array = []
averages_array = []
averages_array1 = []
time_array = []
Q_array = []
for epsilon in [ 0.3,0.5, 0.7,0.8, 0.9, 0.95]:
    start_time = time.time()
    Q = np.zeros((environment.observation_space.n, environment.action_space.n))
    rewards = []
    iterations = []
    optimal = [0] * environment.observation_space.n
    alpha = 0.85
    gamma = 0.95
    episodes = 8000
    environment = 'FrozenLake-v1'
    environment = gym.make(environment)
    environment = environment.unwrapped
    desc = environment.unwrapped.desc
    for episode in range(episodes):
        state = environment.reset()
        state = state[0]
        done = False
        total_reward = 0
        max_steps = 200
        for i in range(max_steps):
            if done:
                break
            current = state
            if np.random.uniform(0, 1) < (epsilon):
                action = environment.action_space.sample()
            else:
                action = np.argmax(Q[current, :])
            state, reward, terminated, truncated, info = environment.step(action)
            done = terminated or truncated
            total_reward += reward
            Q[current, action] += alpha * (reward + gamma * np.max(Q[state, :]) - Q[current, action])
        rewards.append(total_reward)
        iterations.append(i)
    for state in range(environment.observation_space.n):
        optimal[state] = np.argmax(Q[state, :])
    reward_array.append(rewards)
    r1=np.array(rewards)
    r_array.append(np.mean(r1))
    iterations = np.array(iterations)
    iter_array.append(np.sum(iterations)/episodes)
    Q_array.append(Q)
    environment.close()
    end_time = time.time()
    time_array.append(end_time - start_time)

    # Plot results
    def t_list(l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

    size = int(episodes / 50)
    ts = list(t_list(rewards, size))
    averages = [sum(time) / len(time) for time in ts]
    size_array.append(size)
    ts_array.append(ts)
    averages_array.append(averages)
    averages_array1.append(np.mean(averages))

plt.plot([ 0.3,0.5,  0.7, 0.8,0.9, 0.95], time_array)
plt.xlabel('Epsilon')
plt.grid()
plt.title('Frozen Lake - QLearning Analysis: Epsilon')
plt.ylabel('Execution Time (s)')
plt.savefig('images/FL_QL_EP_T.png')
#plt.show()
plt.close()

plt.plot([ 0.3,0.5,  0.7, 0.8,0.9, 0.95], iter_array)
plt.xlabel('Epsilon')
plt.grid()
plt.title('Frozen Lake - QLearning Analysis: Iterations to Converge')
plt.ylabel('Iterations to Converge')
plt.savefig('images/FL_QL_EP_IC.png')
#plt.show()
plt.close()

plt.plot([ 0.3,0.5,  0.7,0.8, 0.9, 0.95], averages_array1)
plt.xlabel('Epsilon ')
plt.grid()
plt.title('Frozen Lake - Q Learning')
plt.ylabel('Average Rewards')
plt.savefig('images/FL_QL_EP_AR.png')
#plt.show()
plt.close()

plt.subplot(1, 4, 1)
plt.imshow(Q_array[0])
plt.title('epsilon=0.1')

plt.subplot(1, 4, 2)
plt.title('epsilon=0.5')
plt.imshow(Q_array[2])

plt.subplot(1, 4, 3)
plt.title('epsilon=0.7')
plt.imshow(Q_array[3])

plt.subplot(1, 4, 4)
plt.title('epsilon=0.9')
plt.imshow(Q_array[4])
plt.colorbar()
plt.savefig('images/FL_QL_EP2.png')
#plt.show()
plt.close()


plt.plot(range(0, len(reward_array[2]), size_array[2]), averages_array[2], label='epsilon=0.5')
plt.plot(range(0, len(reward_array[3]), size_array[3]), averages_array[3], label='epsilon=0.7')
plt.plot(range(0, len(reward_array[5]), size_array[5]), averages_array[5], label='epsilon=0.95')

plt.legend()
plt.xlabel('Epsiodes')
plt.grid()
plt.title('Frozen Lake - Q Learning - Constant epsilon')
plt.ylabel('Average Reward')
plt.savefig('images/FL_QL_EPSILON.png')
#plt.show()
plt.close()

reward_array = []
r_array=[]
iter_array = []
size_array = []
ts_array = []
averages_array = []
averages_array1 = []
time_array = []
Q_array = []
for gamma in [0.1, 0.3,0.5,  0.7, 0.9, 0.95]:
    start_time = time.time()
    Q = np.zeros((environment.observation_space.n, environment.action_space.n))
    rewards = []
    iterations = []
    optimal = [0] * environment.observation_space.n
    alpha = 0.8
    epsilon = 0.3
    episodes = 8000
    environment = 'FrozenLake-v1'
    environment = gym.make(environment)
    environment = environment.unwrapped
    desc = environment.unwrapped.desc
    for episode in range(episodes):
        state = environment.reset()
        state = state[0]
        done = False
        total_reward = 0
        max_steps = 200
        for i in range(max_steps):
            if done:
                break
            current = state
            if np.random.uniform(0, 1) < (epsilon):
                action = environment.action_space.sample()
            else:
                action = np.argmax(Q[current, :])
            state, reward, terminated, truncated, info = environment.step(action)
            done = terminated or truncated
            total_reward += reward
            Q[current, action] += alpha * (reward + gamma * np.max(Q[state, :]) - Q[current, action])
        rewards.append(total_reward)
        iterations.append(i)
    for state in range(environment.observation_space.n):
        optimal[state] = np.argmax(Q[state, :])
    reward_array.append(rewards)
    r1=np.array(rewards)
    r_array.append(np.mean(r1))
    iterations = np.array(iterations)
    iter_array.append(np.sum(iterations)/episodes)
    Q_array.append(Q)
    environment.close()
    end_time = time.time()
    time_array.append(end_time - start_time)

    # Plot results
    def t_list(l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

    size = int(episodes / 50)
    ts = list(t_list(rewards, size))
    averages = [sum(time) / len(time) for time in ts]
    size_array.append(size)
    ts_array.append(ts)
    averages_array.append(averages)
    averages_array1.append(np.mean(averages))

plt.plot([0.1, 0.3,0.5,  0.7, 0.9, 0.95], time_array)
plt.xlabel('Gamma')
plt.grid()
plt.title('Frozen Lake - QLearning Analysis: Gamma')
plt.ylabel('Execution Time (s)')
plt.savefig('images/FL_QL_GM_T.png')
#plt.show()
plt.close()

plt.plot([0.1, 0.3,0.5,  0.7, 0.9, 0.95], iter_array)
plt.xlabel('Gamma')
plt.grid()
plt.title('Frozen Lake - QLearning Analysis: Iterations to Converge')
plt.ylabel('Iterations to Converge')
plt.savefig('images/FL_QL_GM_IC.png')
#plt.show()
plt.close()

plt.plot([0.1, 0.3,0.5,  0.7, 0.9, 0.95], averages_array1)
plt.xlabel('Gamma')
plt.grid()
plt.title('Frozen Lake - QLearning Analysis: Average Rewards')
plt.ylabel('Average Rewards')
plt.savefig('images/FL_QL_GM_AR.png')
#plt.show()
plt.close()

plt.subplot(1, 4, 1)
plt.imshow(Q_array[0])
plt.title('gamma=0.1')

plt.subplot(1, 4, 2)
plt.title('gamma=0.5')
plt.imshow(Q_array[2])

plt.subplot(1, 4, 3)
plt.title('gamma=0.7')
plt.imshow(Q_array[3])

plt.subplot(1, 4, 4)
plt.title('gamma=0.9')
plt.imshow(Q_array[4])
plt.colorbar()
plt.savefig('images/FL_QL_GM2.png')
#plt.show()
plt.close()

plt.plot(range(0, len(reward_array[2]), size_array[2]), averages_array[2], label='gamma=0.5')
plt.plot(range(0, len(reward_array[3]), size_array[3]), averages_array[3], label='gamma=0.7')
plt.plot(range(0, len(reward_array[5]), size_array[5]), averages_array[5], label='gamma=0.95')

plt.legend()
plt.xlabel('Epsiodes')
plt.grid()
plt.title('Frozen Lake - Q Learning - Constant gamma')
plt.ylabel('Average Reward')
plt.savefig('images/FL_QL_GAMMA.png')
#plt.show()
plt.close()

reward_array = []
r_array=[]
iter_array = []
size_array = []
ts_array = []
averages_array = []
averages_array1 = []
time_array = []
Q_array = []
for alpha in [0.1, 0.3,0.5,  0.7, 0.9, 0.95]:
    start_time = time.time()
    Q = np.zeros((environment.observation_space.n, environment.action_space.n))
    rewards = []
    iterations = []
    optimal = [0] * environment.observation_space.n
    gamma = 0.95
    epsilon = 0.3
    episodes = 8000
    environment = 'FrozenLake-v1'
    environment = gym.make(environment)
    environment = environment.unwrapped
    desc = environment.unwrapped.desc
    for episode in range(episodes):
        state = environment.reset()
        state = state[0]
        done = False
        total_reward = 0
        max_steps = 200
        for i in range(max_steps):
            if done:
                break
            current = state
            if np.random.uniform(0, 1) < (epsilon):
                action = environment.action_space.sample()
            else:
                action = np.argmax(Q[current, :])
            state, reward, terminated, truncated, info = environment.step(action)
            done = terminated or truncated
            total_reward += reward
            Q[current, action] += alpha * (reward + gamma * np.max(Q[state, :]) - Q[current, action])
        rewards.append(total_reward)
        iterations.append(i)
    for state in range(environment.observation_space.n):
        optimal[state] = np.argmax(Q[state, :])
    reward_array.append(rewards)
    r1=np.array(rewards)
    r_array.append(np.mean(r1))
    iterations = np.array(iterations)
    iter_array.append(np.sum(iterations)/episodes)
    Q_array.append(Q)
    environment.close()
    end_time = time.time()
    time_array.append(end_time - start_time)

    # Plot results
    def t_list(l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

    size = int(episodes / 50)
    ts = list(t_list(rewards, size))
    averages = [sum(time) / len(time) for time in ts]
    size_array.append(size)
    ts_array.append(ts)
    averages_array.append(averages)
    averages_array1.append(np.mean(averages))

plt.plot([0.1, 0.3,0.5,  0.7, 0.9, 0.95], time_array)
plt.xlabel('Alpha')
plt.grid()
plt.title('Frozen Lake - QLearning Analysis: Alpha')
plt.ylabel('Execution Time (s)')
plt.savefig('images/FL_QL_AP_T.png')
#plt.show()
plt.close()

plt.plot([0.1, 0.3,0.5,  0.7, 0.9, 0.95], iter_array)
plt.xlabel('Alpha')
plt.grid()
plt.title('Frozen Lake - QLearning Analysis: Iterations to Converge')
plt.ylabel('Iterations to Converge')
plt.savefig('images/FL_QL_AP_IC.png')
#plt.show()
plt.close()

plt.plot([0.1, 0.3,0.5,  0.7, 0.9, 0.95], averages_array1)
plt.xlabel('Alpha')
plt.grid()
plt.title('Frozen Lake - QLearning Analysis: Average Rewards')
plt.ylabel('Average Rewards')
plt.savefig('images/FL_QL_AP_AR.png')
#plt.show()
plt.close()

plt.subplot(1, 4, 1)
plt.imshow(Q_array[0])
plt.title('alpha=0.1')

plt.subplot(1, 4, 2)
plt.title('alpha=0.5')
plt.imshow(Q_array[2])

plt.subplot(1, 4, 3)
plt.title('alpha=0.7')
plt.imshow(Q_array[3])

plt.subplot(1, 4, 4)
plt.title('alpha=0.9')
plt.imshow(Q_array[4])
plt.colorbar()
plt.savefig('images/FL_QL_AP2.png')
#plt.show()
plt.close()

plt.plot(range(0, len(reward_array[2]), size_array[2]), averages_array[2], label='alpha=0.5')
plt.plot(range(0, len(reward_array[3]), size_array[3]), averages_array[3], label='alpha=0.7')
plt.plot(range(0, len(reward_array[5]), size_array[5]), averages_array[5], label='alpha=0.95')

plt.legend()
plt.xlabel('Epsiodes')
plt.grid()
plt.title('Frozen Lake - Q Learning - Constant alpha')
plt.ylabel('Average Reward')
plt.savefig('images/FL_QL_ALPHA.png')
#plt.show()
plt.close()

def colors_lake():
    return {
            b'S': 'green',
            b'F': 'skyblue',
            b'H': 'black',
            b'G': 'gold',
                 }


def directions_lake():
    return {
            3: '⬆',
            2: '➡',
            1: '⬇',
            0: '⬅'
        }

def plot_policy_map(title, policy, map_desc, color_map, direction_map, print=False):
    figure = plt.figure()
    ax = figure.add_subplot(111, xlim=(0, policy.shape[1]), ylim=(0, policy.shape[0]))
    if policy.shape[1] > 16:
        font_size = 'small'
    plt.title(title)
    for i in range(policy.shape[0]):
        for j in range(policy.shape[1]):
            y = policy.shape[0] - i - 1
            x = j
            best_policy = plt.Rectangle((x,y), 1, 1)
            best_policy.set_facecolor(color_map[map_desc[i, j]])
            ax.add_patch(best_policy)
    plt.axis('off')
    plt.xlim((0, policy.shape[1]))
    plt.ylim((0, policy.shape[0]))
    plt.tight_layout()
    if print:
        plt.savefig('images/FL_QL_2D.png')
    else:
        plt.show()
    plt.close()
    return (plt)


def get_score(environment, policy, episodes=1000):
    misses = 0
    steps_list = []
    for episode in range(episodes):
        observation = environment.reset()
        observation = observation[0]
        steps = 0
        while True:
            action = policy[observation]
            observation, reward, terminated, truncated, info = environment.step(action)
            done = terminated or truncated
            steps += 1
            if done and reward == 1:
                steps_list.append(steps)
                break
            elif done and reward == 0:
                misses += 1
                break
    print('Average number of steps taken, {:.0f} steps'.format(np.mean(steps_list)))
    print('Fall in the hole {:.2f} % of the times'.format((misses / episodes) * 100))


environment = 'FrozenLake-v1'   #16 states in grid world mdp
environment = gym.make(environment)
environment = environment.unwrapped
desc = environment.unwrapped.desc

Q = np.zeros((environment.observation_space.n, environment.action_space.n))
Q1 = np.zeros((environment.observation_space.n, environment.action_space.n))
times=0
alpha = 0.15
gamma = 0.95
epsilon = 0.3
episodes = 12000
rewards=[]
iterations=[]
start_time = time.time()
score1=0
t1=[]
for episode in range(episodes):
    s1=time.time()
    x1=0
    state = environment.reset()
    state= state[0]
    done = False
    total_reward = 0
    max_steps = 200
    new_policy = np.empty(16)
    for i in range(max_steps):
        if done:
            break
        current=state
        if np.random.uniform(0,1) < (epsilon):
            action = environment.action_space.sample()
        else:
            action = np.argmax(Q[current, :])
        state, reward, terminated, truncated, info = environment.step(action)
        done = terminated or truncated
        total_reward += reward
        Q[current, action] += alpha * (reward + gamma * np.max(Q[state, :]) - Q[current, action])
    rewards.append(total_reward)
    iterations.append(i)
    e1=time.time()
    t1.append(e1-s1)

print("Average Rewards:", np.average(rewards))
environment.close()
end_time = time.time()
print("Time in Seconds :", end_time - start_time)

def chunk_list(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]
size = 100
chunks = list(chunk_list(rewards, size))
averages = [sum(chunk) / len(chunk) for chunk in chunks]

plt.plot(range(0, len(rewards), size), averages)
plt.xlabel('Iterations')
plt.ylabel('Average Reward')
plt.title('Frozen Lake - QLearning Analysis: Average Rewards vs Iterations')
plt.grid()
#plt.show()
plt.savefig('images/FL_QL_AR.png')
plt.close()

plt.title('Frozen Lake - Q value array')
plt.imshow(Q)
plt.colorbar()
#plt.show()
plt.savefig('images/FL_QL_QA.png')
plt.close()

policy=np.empty(16)
for state in range(16):
    policy[state] = np.argmax(Q[state, :])
scores = evaluate_policy(environment, policy, 0.95)
print("Final Reward: ",scores)
plot = plot_policy_map('Frozen Lake - Policy Map Result: QLearning ' + 'Gamma: ' + str(gamma), policy.reshape(4, 4), desc, colors_lake(), directions_lake(),True)
print(t1[10])
get_score(environment,policy)