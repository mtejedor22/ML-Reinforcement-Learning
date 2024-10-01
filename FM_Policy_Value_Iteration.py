import matplotlib.pyplot as plt
import numpy as np
import hiive.mdptoolbox.mdp,hiive.mdptoolbox.example


P,R=hiive.mdptoolbox.example.forest(S=500)
times=np.empty(10)
discount_factor=np.empty(10)
iteration=np.empty(10)
score=np.empty(10)

for i in range(0, 10):
    policy_iteration = hiive.mdptoolbox.mdp.PolicyIteration(P, R, (i + 0.5) / 10)
    policy_iteration.setVerbose()
    policy_iteration.run()
    discount_factor[i] = (i + 0.5) / 10
    score[i] = np.mean(policy_iteration.V)
    iteration[i] = policy_iteration.iter
    times[i] = policy_iteration.time
    best_policy=policy_iteration.policy
    best_policy=np.array(best_policy)


plt.plot(discount_factor, times)
plt.xlabel('Discount factor')
plt.title('Forest Management - Policy Iteration Analysis: Discount Factor')
plt.ylabel(' Time taken (in seconds)')
plt.grid(True)
plt.savefig('images/FM_PI_DF.png')
#plt.show()
plt.close()

plt.plot(discount_factor, score)
plt.xlabel('Discount factor')
plt.ylabel('Average Rewards')
plt.title('Forest Management - Policy Iteration Analysis: Average Rewards')
plt.grid(True)
plt.savefig('images/FM_PI_AR.png')
#plt.show()
plt.close() 

plt.plot(discount_factor, iteration)
plt.xlabel('Discount factor')
plt.ylabel('Iterations to Converge')
plt.title('Forest Management - Policy Iteration Analysis: Iterations to Converge')
plt.grid(True)
plt.savefig('images/FM_PI_IC.png')
#plt.show()
plt.close()  

P, R = hiive.mdptoolbox.example.forest(S=500)
times=np.empty(10)
discount_factor=np.empty(10)
iteration=np.empty(10)
score=np.empty(10)

for i in range(0, 10):
    policy_iteration = hiive.mdptoolbox.mdp.ValueIteration(P, R, (i + 0.5) / 10)
    policy_iteration.setVerbose()
    policy_iteration.run()
    discount_factor[i] = (i + 0.5) / 10
    score[i] = np.mean(policy_iteration.V)
    iteration[i] = policy_iteration.iter
    times[i] = policy_iteration.time
    best_policy=policy_iteration.policy
    best_policy=np.array(best_policy)


plt.plot(discount_factor, times)
plt.xlabel('Discount factor')
plt.title('Forest Management - Value Iteration Analysis: Discount Factor')
plt.ylabel('Execution Time (in seconds)')
plt.grid(True)
plt.savefig('images/FM_VI_DF.png')
#plt.show()
plt.close() 


plt.plot(discount_factor, score)
plt.xlabel('Discount factor')
plt.ylabel('Average Rewards')
plt.title('Forest Management - Value Iteration Analysis: Average Rewards')
plt.grid(True)
plt.savefig('images/FM_VI_AR.png')
#plt.show()
plt.close() 

plt.plot(discount_factor, iteration)
plt.xlabel('Discount factor')
plt.ylabel('Iterations to Converge')
plt.title('Forest Management - Value Iteration Analysis: Iterations to Converge')
plt.grid(True)
plt.savefig('images/FM_VI_IC.png')
#plt.show()
plt.close() 

policy_iteration = hiive.mdptoolbox.mdp.PolicyIteration(P, R, 0.95)
policy_iteration.run()
print("Policy Iteration Score: ",np.mean(policy_iteration.V))
print("Number of Iterations: ",policy_iteration.iter)
print("Time in Seconds: ",policy_iteration.time)
best_policy = policy_iteration.policy
best_policy = np.array(best_policy)
print("Best Policy")
print(best_policy.reshape(10, 50))

policy_iteration = hiive.mdptoolbox.mdp.ValueIteration(P, R, 0.95)
policy_iteration.run()
print("Value Iteration Score: ",np.mean(policy_iteration.V))
print("Number of Iterations: ",policy_iteration.iter)
print("Time in Seconds: ",policy_iteration.time)
best_policy = policy_iteration.policy
best_policy = np.array(best_policy)
print("Best Policy")
print(best_policy.reshape(10, 50))




