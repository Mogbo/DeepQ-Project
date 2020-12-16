import re
import numpy as np
import matplotlib.pyplot as plt
import glob

def process_file(fname):
    rewards = []

    with open(fname, 'r') as fh:
        all_lines = fh.readlines()

    for line in all_lines:
        if ("Episode reward") in line:
            match = re.search(r"reward: (\d+\.\d+)", line)
            reward = float( match.group(1))
            rewards.append(reward)

    return np.array(rewards)

rewards_mdqn = process_file("MDQN_cartpole_52.txt")
rewards_ddqn = process_file("DDQN_cartpole_52.txt")
rewards_dqn = process_file("DQN_cartpole_52.txt")
#rewards_norm_done = np.array(process_file("checkpoints/CartPole-v0/100k_openai/DDQN/log.txt"))
#rewards_ddqn = np.array(process_file("checkpoints/CartPole-v0/100k_openai/DDQN/log.txt"))
# rewards_not_norm_done = np.array(process_file("not_norm_done.txt"))

# print("Norm:", np.array(rewards_norm_done)[700:].mean())
# print("Not norm:", np.array(rewards_not_norm_done)[700:].mean())

#plt.plot(np.arange(1, len(rewards_mdqn) + 1), rewards_mdqn, label="mdqn")
#plt.plot(np.arange(1, len(rewards_ddqn) + 1), rewards_ddqn, label="ddqn")

running_avg_mdqn = []
for idx in range(100, len(rewards_mdqn)):
    running_avg_mdqn.append(rewards_mdqn[idx - 100: idx].mean())
running_avg_mdqn = np.array(running_avg_mdqn)

running_avg_ddqn = []
for idx in range(100, len(rewards_ddqn)):
    running_avg_ddqn.append(rewards_ddqn[idx - 100: idx].mean())
running_avg_ddqn = np.array(running_avg_ddqn)

running_avg_dqn = []
for idx in range(100, len(rewards_dqn)):
    running_avg_dqn.append(rewards_dqn[idx - 100: idx].mean())
running_avg_dqn = np.array(running_avg_dqn)

plt.plot(np.arange(100, len(rewards_mdqn)), running_avg_mdqn, label="MDQN", color='C0')
plt.plot(np.arange(100, len(rewards_ddqn)), running_avg_ddqn, label="DDQN", color='C1')
plt.plot(np.arange(100, len(rewards_dqn)), running_avg_dqn, label="DQN", color='C2')

plt.axhline(running_avg_mdqn.mean(), label="MDQN avg", color='C0', ls = '--')
plt.axhline(running_avg_ddqn.mean(), label="DDQN avg", color='C1', ls = '--')
plt.axhline(running_avg_dqn.mean(), label="DQN avg", color='C2', ls = '--')

plt.xlabel("Episode")
plt.ylabel("Running Average Reward (100 steps)")
plt.axhline(195, label="target", color='r')
plt.legend()