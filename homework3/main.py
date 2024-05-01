from abc import ABC, abstractmethod
import logging
import os
import pickle
import random
from sqlite3 import paramstyle
from tkinter import E

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger.info("Logging started")

class Gridworld():
    def __init__(self, start=36, row=4, column=12):
        # 4 * 12 gridworld
        self.row = row
        self.column = column
        self.size = self.row*self.column
        # by the question
        self.goal = 47
        self.cliff_reward = -100

        if start == self.goal:
            raise ValueError("Start cannot be the goal")
        self.start = start
        self.state = self.start

        self.action = ['n', 's', 'w', 'e']
        self.action_prob = {'n': 0.25, 's': 0.25, 'w': 0.25, 'e': 0.25}
        logger.info("Gridworld is created")

    def get_next_state(self,state,action):
        # ensure legal action
        if (0<=state<=11 and action == 'n') or\
                (36<=state<=47 and action == 's') or\
                (state%12 == 0 and action == 'w') or\
                (state in [11,23,35] and action == 'e'):
            return state, self.get_reward(state)
        
        # update legal action
        if action == 'n':
            s_prime = state - self.column
        elif action == 's':
            s_prime = state + self.column
        elif action == 'w':
            s_prime = state - 1
        elif action == 'e':
            s_prime = state + 1
        
        if 37<=s_prime<=46:
            return self.start, self.get_reward(self.start,cliff=True)
        return s_prime, self.get_reward(s_prime)
    def get_reward(self,state,cliff=False):
        if state == self.goal:
            return 0
        if 37<=state<=46 or cliff:
            return -100
        return -1
    def move(self,state,action):
        s_prime = self.get_next_state(state,action)
        self.state = s_prime
    def next_step(self, action):
        self.state, _ = self.get_next_state(self.state,action=action)
        return self.state
    def next_step_and_reward(self,action):
        self.state, reward = self.get_next_state(self.state,action)
        return self.state, reward
    def is_finish(self):
        return self.state == self.goal
    def reset(self):
        self.state = self.start
        return self.state

class Agent(ABC):
    def __init__(self, grid, gamma=0.9, alpha=0.2, epsilon=0.1, max_iteration=1000):
        self.grid = grid
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.decay_value = 0.999
        self.loading_view = False
        
        self.max_iteration = max_iteration
        self.run_limit = 100
        # quality of state, action pair
        self.quality = {(i,a):0 for i in range(grid.size) for a in grid.action}
        logger.info(f"{self.__class__.__name__} is created")
    def loading_animation(self,disable=False):
        self.loading_view = disable
    def decay(self):
        if self.epsilon<0.1:
            return
        self.epsilon *= self.decay_value
    def clear_quality(self):
        self.quality = {(i,a):0 for i in range(self.grid.size) for a in self.grid.action}
    @abstractmethod
    def get_optimal_action(self):
        pass
    @abstractmethod
    def train(self):
        pass
    def load_pickle(self, filename):
        import pickle
        with open(filename, 'rb') as f:
            self.value = pickle.load(f)
    def move(self):
        action = self.get_optimal_action()
        s_prime,r = self.grid.next_step_and_reward(action)
        return action,r
    def start(self):
        logger.info(f"The starting point is {self.grid.start}")
        logger.info(f"This {self.__class__.__name__} will start now.")
        self.grid.reset()
        action = []
        policy = None
        reward = 0
        while not self.grid.is_finish() and len(action) < self.run_limit:
            text = f"state was {self.grid.state}, "
            policy,r = self.move()
            logger.debug(text + f"action is {policy}, state is now {self.grid.state}")
            action.append(policy)
            reward += r
        logger.info(f"Number of steps taken is {len(action)}, reward is {reward}")
        logger.debug(f"The action taken are {action}")
        logger.info(f"==================================================")
        return reward
    def print_agent_table(self):
        print(self.quality)
    def print_params(self):
        print(f"alpha: {self.alpha}, gamma: {self.gamma}, epsilon: {self.epsilon}")
class QLearning(Agent):
    def __init__(self, grid, gamma=0.9, alpha=0.2, epsilon=0.1, max_iteration=1000):
        super().__init__(grid, gamma, alpha, epsilon, max_iteration)
    def train(self):
        for _ in tqdm(range(self.max_iteration), disable=self.loading_view):
            self.grid.reset()
            steps = 0
            while not self.grid.is_finish() and steps < self.run_limit:
                s = self.grid.state
                action = self.get_action()
                s_prime, reward = self.grid.next_step_and_reward(action)
                self.quality[(s,action)] += \
                        self.alpha * (
                            reward + \
                            self.gamma * max([self.quality[(s_prime,a)] for a in self.grid.action])\
                            - self.quality[(s,action)]
                        )
                steps+=1
    def get_optimal_action(self):
        s = self.grid.state
        return max(self.grid.action, key=lambda a: self.quality[(s,a)])
    def get_action(self):
        if random.random() < self.epsilon:
            action = random.choice(self.grid.action)
        else:
            action = self.get_optimal_action()
        return action
class SARSA(Agent):
    def __init__(self, grid, gamma=0.9, alpha=0.2, epsilon=0.1, max_iteration=1000):
        super().__init__(grid, gamma, alpha, epsilon, max_iteration)
    def train(self):
        for _ in tqdm(range(self.max_iteration),disable=self.loading_view):
            self.grid.reset()
            s = self.grid.state
            action = self.get_action()
            steps = 0
            while not self.grid.is_finish() and steps < self.run_limit:
                s_prime, reward = self.grid.next_step_and_reward(action)
                # on policy learning
                action_prime = self.get_action()
                self.quality[(s,action)] += \
                        self.alpha * (
                            reward + \
                            self.gamma * self.quality[(s_prime,action_prime)]\
                            - self.quality[(s,action)]
                        )
                action = action_prime
                s = s_prime
                steps+=1
    def get_optimal_action(self):
        return max(self.grid.action,
                   key=lambda a: self.quality[(self.grid.state,a)])
    def get_action(self):
        if random.random() < self.epsilon:
            action = random.choice(self.grid.action)
        else:
            action = self.get_optimal_action()
        return action
    
class Experiment():
    def __init__(self, grid, agent=None, override=True,
                 load_agent=False, agent_list=[]):
        self.grid = grid
        self.agent = agent
        if not load_agent and agent is None and agent_list == []:
            raise Exception("Agent is not provided")
        if agent_list != []:
            self.agent_list = agent_list
            return
        # store the pickle file
        pickle_file = os.path.join(os.path.dirname(__file__),f"{self.agent.__class__.__name__}.pkl")
        if os.path.exists(pickle_file) and override == False:
            with open(pickle_file, 'rb') as file:
                self.agent = pickle.load(file)
                self.agent.grid = self.grid
        else:
            self.agent.train()
            with open(pickle_file, 'wb') as file:
                pickle.dump(self.agent,file)
        logger.info("Experiment created")
    def run(self):
        return self.agent.start()
    def train(self, max_iteration=10, save_agent=False):
        self.agent.max_iteration = max_iteration
        self.agent.train()
        if save_agent:
            with open(f"{self.agent.__class__.__name__}.pkl", 'wb') as file:
                    pickle.dump(self.agent,file)
    def set_agent(self, agent):
        self.agent = agent
    def disable_all_loading(self):
        for agent in self.agent_list:
            agent.loading_animation(True)
    def train_agent(self, iteration, params:tuple):
        for agent in self.agent_list:
            agent.alpha, agent.gamma, agent.epsilon = params
            agent.clear_quality()
        
        reward_list = {agent.__class__.__name__:[] for agent in self.agent_list}
        for _ in range(iteration):
            for agent in self.agent_list:
                self.set_agent(agent)
                self.train(1)
                r = self.run()
                reward_list[self.agent.__class__.__name__].append(max(r,self.grid.cliff_reward))
        return reward_list
    def plot_performance(self, iteration, params):
        if self.agent_list == []:
            return
        
        reward_list = dict()
        for param in params:
            reward_list[str(param)] = self.train_agent(iteration, param)
        
        
        df = pd.DataFrame(reward_list)
        fig, axs = plt.subplots(2,3,figsize=(15,7))
        for i in range(2):
            for j in range(3):
                param = params[i*3+j]
                axs[i,j].plot(df[str(param)].QLearning, label="QLearning")
                axs[i,j].plot(df[str(param)].SARSA, label="SARSA")
                axs[i,j].set_title(f"alpha: {param[1]}, gamma: {param[0]}, epsilon: {param[2]}")
                axs[i,j].legend()
                axs[i,j].axhline(y=-12, color='r', linestyle='-')
        for ax in axs.flat:
            ax.set(xlabel='Episode', ylabel='Reward')
            ax.label_outer()
        plt.legend()
        plt.show()
        
if __name__ == "__main__":
    logger.info("Start the experiment")
    logger.debug("test")
    grid = Gridworld()
    sarsa = SARSA(grid)
    qlearning = QLearning(grid)
    agents = [sarsa,qlearning]
    # gamma=0.9, alpha=0.2, epsilon=0.1
    params = [(0.9, 0.1, 0.0000001),(0.9, 0.2, 0.1),(0.9, 0.2, 0.1)
              ,(0.9, 0.1, 0.2),(0.9, 0.2, 0.3),(0.9, 0.2, 0.2)]
    e = Experiment(grid,override=True,agent_list=agents)
    e.disable_all_loading()
    e.plot_performance(1000,params)
    logger.info("End the experiment")