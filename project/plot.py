import argparse
import os

import pandas as pd

from matplotlib import pyplot as plt

import numpy as np

pre = argparse.ArgumentParser()
pre.add_argument('--experiment', type=str,
                 choices=['atari', 'mujoco'],
                 default='mujoco',
                 help='exp to run')
pre.add_argument('--env',  type=str,
                 choices=['Ant-v4', 'Hopper-v4', 'HalfCheetah-v4', 'Humanoid-v4','Breakout','Pong'],
                 default='Ant-v4',
                 help='env to run')
args_pre, extra = pre.parse_known_args()

env_name = args_pre.env

# set agent name for the files
if args_pre.experiment == 'atari':
    agents = ["DQN", "DDQN"]
else:
    agents = ["PPO", "SAC"]


files = [f"completed/{env_name}_{agents[0]}.csv", f"completed/{env_name}_{agents[1]}.csv"]
df = list()

for file in files:
    # check if the csv file exists
    if not os.path.exists(file):
        print(f"{file} does not exist")
        raise Exception(f"File {file} does not exist")

    df.append(pd.read_csv(file, header=0, sep='\s*,\s*'))

if args_pre.experiment == 'atari':
    for i in range(2):
        plt.plot(df[i]['step'], df[i]['avg_reward'], label=f'{agents[i]}')
        # Fill the region between the upper and lower bounds of the standard deviation
        plt.fill_between(df[i]['step'], np.subtract(df[i]['avg_reward'], df[i]['std']), np.add(df[i]['avg_reward'], df[i]['std']), alpha=0.2)
else:
    for i in range(2):
        plt.plot(df[i]['steps'], df[i]['mean_return'], label=f'{agents[i]}')
        # Fill the region between the upper and lower bounds of the standard deviation
        plt.fill_between(df[i]['steps'], np.subtract(df[i]['mean_return'], df[i]['std']), np.add(df[i]['mean_return'], df[i]['std']), alpha=0.2)

plt.xlabel('Steps')
plt.ylabel('Mean Return')
plt.title(f'{env_name}')
plt.legend()
plt.show()