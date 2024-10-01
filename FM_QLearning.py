import matplotlib.pyplot as plt
import numpy as np
import time
import hiive.mdptoolbox.mdp,hiive.mdptoolbox.example

P, R = hiive.mdptoolbox.example.forest(S=500)
avg_value_states = []
Q_table = []
policy = []
iterations = []
time_array = []
alpha_array= [ 0.1, 0.3, 0.5, 0.7,0.85, 0.95]
for alpha in alpha_array:
    start_time = time.time()
    ql = hiive.mdptoolbox.mdp.QLearning(P, R,0.95,epsilon=0.7,n_iter=15000,alpha=alpha)
    ql.run()
    end_time = time.time()
    avg_value_states.append(np.mean(ql.V))
    policy.append(ql.policy)
    time_array.append(end_time - start_time)
    Q_table.append(ql.Q)

figure, axs = plt.subplots(1, 2, figsize=(12, 4))

# Plot 1: Average value of the states vs Learning Rate
axs[0].plot(alpha_array, avg_value_states)
axs[0].set_title('Average value of the states vs Learning Rate')
axs[0].set_ylabel('Average value of the states')
axs[0].set_xlabel('Learning Rate')
axs[0].grid()

# Plot 2: Time vs Learning Rate
axs[1].plot(alpha_array, time_array)
axs[1].set_title('Time vs Learning Rate')
axs[1].set_ylabel('Time (s)')
axs[1].set_xlabel('Learning Rate')
axs[1].grid()

# Add a general title for the entire plot
figure.suptitle('Forest Management - QLearning Analysis')

# Save the figure
plt.savefig('images/FM_QL_LR.png')

# Close the plot
plt.close()

P, R = hiive.mdptoolbox.example.forest(S=500)
avg_value_states = []
policy = []
iterations = []
time_array = []
Q_table = []
epsilon_array = [ 0.1, 0.3, 0.5, 0.7,0.85, 0.95]
for epsilon in epsilon_array:
    start_time = time.time()
    ql = hiive.mdptoolbox.mdp.QLearning(P, R,0.95,epsilon=epsilon,n_iter=15000)
    ql.run()
    end_time = time.time()
    avg_value_states.append(np.mean(ql.V))
    policy.append(ql.policy)
    time_array.append(end_time - start_time)
    Q_table.append(ql.Q)

figure, axs = plt.subplots(1, 2, figsize=(12, 4))  # 1 row, 2 columns

# Plot 1: Average value of the states vs Epsilon
axs[0].plot(epsilon_array, avg_value_states)
axs[0].set_title('Average value of the states vs Epsilon')
axs[0].set_ylabel('Average value of the states')
axs[0].set_xlabel('Epsilon')
axs[0].grid()

# Plot 2: Time vs Epsilon
axs[1].plot(epsilon_array, time_array)
axs[1].set_title('Time vs Epsilon')
axs[1].set_ylabel('Time (s)')
axs[1].set_xlabel('Epsilon')
axs[1].grid()

# Add a general title for the entire plot
figure.suptitle('Forest Management - QLearning Analysis')

# Save the figure
plt.savefig('images/FM_QL_EP.png')

# Close the plot
plt.close()


P, R = hiive.mdptoolbox.example.forest(S=500)
avg_value_states = []
policy = []
iterations = []
time_array = []
Q_table = []
gamma_array = [0.1, 0.3, 0.5, 0.7,0.85, 0.95]

for gamma in gamma_array:
    start_time = time.time()
    ql = hiive.mdptoolbox.mdp.QLearning(P, R,gamma=gamma,epsilon=0.85,n_iter=15000)
    ql.run()
    end_time = time.time()
    avg_value_states.append(np.mean(ql.V))
    policy.append(ql.policy)
    time_array.append(end_time - start_time)
    Q_table.append(ql.Q)

# Create subplots
figure, axs = plt.subplots(1, 2, figsize=(12, 4))  # 1 row, 2 columns

# Plot 1: Average value of the states vs Gamma
axs[0].plot(gamma_array, avg_value_states)
axs[0].set_title('Average value of the states vs Gamma')
axs[0].set_ylabel('Average value of the states')
axs[0].set_xlabel('Gamma')
axs[0].grid()

# Plot 2: Time vs Gamma
axs[1].plot(gamma_array, time_array)
axs[1].set_title('Time vs Gamma')
axs[1].set_ylabel('Time (s)')
axs[1].set_xlabel('Gamma')
axs[1].grid()

# Add a general title for the entire plot
figure.suptitle('Forest Management - QLearning Analysis')

# Save the figure
plt.savefig('images/FM_QL_GM.png')

# Close the plot
plt.close()

avg_value_states = []
policy = []
time_array = []
Q_table = []
states_array = [300, 500, 1000,1500,2000,2500,3000]
for St in states_array:
    P, R = hiive.mdptoolbox.example.forest(S=St)
    start_time = time.time()
    ql = hiive.mdptoolbox.mdp.QLearning(P, R,0.95,epsilon=0.8,n_iter=15000,alpha=0.95)
    ql.run()
    end_time = time.time()
    avg_value_states.append(np.mean(ql.V))
    time_array.append(end_time - start_time)
    Q_table.append(ql.Q)

# Create subplots
figure, axs = plt.subplots(1, 2, figsize=(12, 4))  # 1 row, 2 columns
# Plot 1: Average value of the states vs Number of States
axs[0].plot(states_array, avg_value_states)
axs[0].set_title('Average value of the states vs Number of States')
axs[0].set_ylabel('Average value of the states')
axs[0].set_xlabel('Number of States')
axs[0].grid()

# Plot 2: Time vs Number of States
axs[1].plot(states_array, time_array)
axs[1].set_title('Time vs Number of States')
axs[1].set_ylabel('Time (s)')
axs[1].set_xlabel('Number of States')
axs[1].grid()

figure.suptitle('Forest Management - QLearning Analysis')

# Save the figure
plt.savefig('images/FM_QL_ST.png')

# Close the plot
plt.close()


avg_value_states = []
policy = []
iterations = []
time_array = []
Q_table = []

probability_array = [0.01,0.05,0.1,0.2,0.3,0.5]

for p1 in probability_array:
    P, R = hiive.mdptoolbox.example.forest(S=500,p=p1)
    start_time = time.time()
    ql = hiive.mdptoolbox.mdp.QLearning(P, R,0.95,epsilon=0.8,n_iter=10000,alpha=0.95)
    ql.run()
    end_time = time.time()
    avg_value_states.append(np.mean(ql.V))
    time_array.append(end_time - start_time)
    Q_table.append(ql.Q)

# Create subplots
figure, axs = plt.subplots(1, 2, figsize=(12, 4))  # 1 row, 2 columns

# Plot 1: Average value of the states vs Fire probability
axs[0].plot(probability_array, avg_value_states)
axs[0].set_title('Average value of the states vs Fire probability')
axs[0].set_ylabel('Average value of the states')
axs[0].set_xlabel('Fire probability')
axs[0].grid()

# Plot 2: Time vs Fire probability
axs[1].plot(probability_array, time_array)
axs[1].set_title('Time vs Fire probability')
axs[1].set_ylabel('Time (s)')
axs[1].set_xlabel('Fire probability')
axs[1].grid()

# Add a general title for the entire plot
figure.suptitle('Forest Management - QLearning Analysis')

# Save the figure
plt.savefig('images/FM_QL_FP.png')

# Close the plot
plt.close()


P, R = hiive.mdptoolbox.example.forest(S=500, p=0.1)
start_time = time.time()
ql = hiive.mdptoolbox.mdp.QLearning(P, R, 0.95, epsilon=0.8, n_iter=100000,epsilon_decay=1,alpha=0.95)
ql.run()
end_time = time.time()
print("Average Value Reward: ",np.mean(ql.V))
print('Time: ',end_time-start_time)
print("First 10 States: ", ql.policy[0:10])
print("Last 10 States: ", ql.policy[490:500])

plt.imshow(ql.Q[:10,:])
plt.title('Forest Management - Q value array')
plt.colorbar()
plt.xlabel('Actions')
plt.ylabel('10 first States')
#plt.show()
plt.savefig('images/FM_QL_QA.png')
plt.close() 