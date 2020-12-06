import re
import numpy as np
import matplotlib.pyplot as plt

def process_file(fname):
    rewards = []

    with open(fname, 'r') as fh:
        all_lines = fh.readlines()

    for line in all_lines:
        if ("Episode reward") in line:
            match = re.search(r"reward: (\d+\.\d+)", line)
            reward = float( match.group(1))
            rewards.append(reward)

    return rewards

rewards_norm_done = np.array(process_file("norm_done.txt"))
rewards_not_norm_done = np.array(process_file("not_norm_done.txt"))

print("Norm:", np.array(rewards_norm_done)[700:].mean())
print("Not norm:", np.array(rewards_not_norm_done)[700:].mean())

plt.plot(np.arange(1, len(rewards_norm_done) + 1), rewards_norm_done, label="raw")
plt.xlabel("Episode")
plt.ylabel("Reward")

running_avg = []

for idx in range(100, len(rewards_norm_done)):
    running_avg.append(rewards_norm_done[idx-100: idx].mean())
running_avg = np.array(running_avg)

plt.plot(np.arange(100, len(rewards_norm_done)), running_avg, marker=".", label="running_avg(100 steps)")
plt.axhline(475, label="target", color='g')
plt.legend()