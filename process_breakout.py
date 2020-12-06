import re
import numpy as np
import matplotlib.pyplot as plt


def process_file(fname):
    rewards = []
    episodes = []

    with open(fname, 'r') as fh:
        all_lines = fh.readlines()

    for line in all_lines:
        if ("running reward") in line:
            match = re.search(r"reward: (\d+\.\d+)", line)
            reward = float(match.group(1))
            rewards.append(reward)

            match2 = re.search(r"episode (\d+)", line)
            episode = float(match2.group(1))
            episodes.append(episode)

    return np.array(episodes), np.array(rewards)

episodes, rewards = process_file("keras_baseline_breakoutv4.log")

plt.plot(episodes, rewards, label="running_avg")
plt.xlabel("Episode")
plt.ylabel("Running Average Reward (100 steps)")
plt.axhline(40, marker="_", color='g', label="target")
plt.legend()
