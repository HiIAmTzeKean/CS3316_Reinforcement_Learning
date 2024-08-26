import os
import argparse
import torch
import gymnasium as gym
from mujoco_exp.agent import PPO, SAC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_STEPS = 1000000


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default='Hopper-v4')
    parser.add_argument('--method', choices=['PPO', 'SAC'], default='SAC')
    args, extra = parser.parse_known_args()
    return args


# if __name__ == '__main__':
args = parse_arguments()
env = gym.make(args.env_name)
if not os.path.exists("models"):
        os.makedirs("models")
if not os.path.exists("results"):
    os.makedirs("results")
local_dir = os.path.join('models', args.env_name, f'{args.method}')

if args.method == 'SAC':
    agent = SAC(env, True, True, NUM_STEPS, local_dir, args.env_name)
else:
    agent = PPO(env, NUM_STEPS, local_dir, args.env_name)
agent.run()
