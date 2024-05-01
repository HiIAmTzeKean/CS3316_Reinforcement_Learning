from abc import ABC
import logging
import os
import pickle
import random
from re import A
import traceback

import pandas as pd

from matplotlib import pyplot as plt
from tqdm import tqdm
from collections import defaultdict 


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

class Gridworld():
    def __init__(self, start=0):
        self.size = 36
        self.goal = [35,1]
        if start in self.goal:
            raise ValueError("Start cannot be the goal")
        self.start = start
        self.state = self.start
        self.action = ['n', 's', 'w', 'e']
        self.action_prob = {'n': 0.25, 's': 0.25, 'w': 0.25, 'e': 0.25}
        print("Gridworld is created")
    def get_start(self):
        return self.start
    def get_size(self):
        return self.size
    def get_reward(self):
        if self.state in self.goal:
            return 0
        return -1
    def get_state(self):
        return self.state
    def reset(self):
        self.state = self.start
        return self.state
    def random_start(self):
        self.start = 1
        while self.start in self.goal:
            self.start = random.randint(0,35)
        self.state = self.start
        return self.state
    def is_finish(self):
        return self.state in self.goal
    def get_next_state(self, action):
        # ensure legal action
        if (0<=self.state<=5 and action == 'n') or\
                (30<=self.state<=35 and action == 's') or\
                (self.state%6 == 0 and action == 'w') or\
                (self.state in [5,11,17,23,29,35] and action == 'e'):
            return self.state
        
        # update legal action
        if action == 'n':
            return self.state - 6
        elif action == 's':
            return self.state + 6
        elif action == 'w':
            return self.state - 1
        elif action == 'e':
            return self.state + 1
    def get_goals(self):
        return self.goal
    def get_training_state(self):
        unvisited = [i for i in range(self.size)]
        for i in self.goal:
            unvisited.remove(i)
        # unvisited.remove(self.start)
        # unvisited.insert(0,self.start)
        return unvisited
    def update_state(self, action):
        self.state = self.get_next_state(action)
        return self.state
    def transition(self,state,action):
        self.state = state
        s_prime = self.get_next_state(action)
        reward = self.get_reward()
        return s_prime, reward

class Agent(ABC):
    def __init__(self, grid, gamma=0.9, max_iteration=1000):
        self.grid = grid
        self.gamma = gamma
        self.reward = 0
        self.done = False
        self.reward = 0
        self.max_iteration = max_iteration
        self.run_limit = 100
        print("Agent is created")

    def get_optimal_action(self):
        pass
    def get_random_action(self):
        return self.grid.action[random.randint(0,3)]
    
    def train(self):
        pass
    
    def load_pickle(self, filename):
        import pickle
        with open(filename, 'rb') as f:
            self.value = pickle.load(f)
    
    def move(self):
        action = self.get_optimal_action()
        self.grid.update_state(action)
        return action
    
    def move_action_defined(self,action):
        self.grid.update_state(action)
        return action

    def reset(self):
        self.state = self.grid.start
        self.reward = 0
        
    def get_episode(self):
        self.grid.reset()
        epoch = []
        while not self.grid.is_finish():
            s = self.grid.get_state()
            policy = self.get_random_action()
            self.grid.update_state(policy)
            s_prime = self.grid.get_state()
            epoch.append((s, policy, self.grid.get_reward(),s_prime))
        return epoch
    def update_grid(self, grid):
        self.grid = grid
        self.grid.reset()
    def start(self):
        logger.debug(f"The starting point is {self.grid.start}")
        logger.debug(f"This {self.__class__.__name__} will start now.")
        self.grid.reset()
        action = []
        policy = None
        while not self.grid.is_finish() and len(action) < self.run_limit:
            logging.info(f"state was {self.grid.get_state()}",end=', ')
            policy = self.move()
            logger.info(f"action is {policy}, state is now {self.grid.get_state()}")
            action.append(policy)
        logger.debug(f"Number of steps taken is {len(action)}")
        logger.debug(f"The action taken are {action}")
        logger.debug(f"==================================================")
        return len(action)

class RandomAgent(Agent):
    def get_optimal_action(self):
        return self.grid.action[random.randint(0,3)]
    
class FirstVisitMonteCarlo(Agent):
    def __init__(self, grid, gamma=0.9):
        super().__init__(grid, gamma)
        print(f"This is agent {self.__class__.__name__}")
        self.value = [0 for i in range(self.grid.get_size())]
        self.returns = [[0,0] for i in range(self.grid.get_size())] # [G, N]
        self.policy = [self.grid.action[random.randint(0,3)]\
                       for i in range(self.grid.get_size())]
    def get_optimal_action(self):
        values = []
        for action in self.grid.action:
            values.append(self.value[self.grid.get_next_state(action)])
        return self.grid.action[values.index(max(values))]  
    def train(self):
        for _ in tqdm(range(self.max_iteration)):
            epoch = self.get_episode()
            G = 0
            visited = set(i for i in self.grid.get_goals())
            for state,action,reward,_ in reversed(epoch):
                self.move_action_defined(action)
                G = self.gamma * G + reward
                if state not in visited:
                    visited.add(state)
                    # update return value
                    self.returns[state][0] = (self.returns[state][0] * self.returns[state][1] + G) / (self.returns[state][1] + 1)
                    self.returns[state][1] += 1
                    self.value[state] = self.returns[state][0]
                    
class EveryVisitMonteCarlo(FirstVisitMonteCarlo):
    def train(self):
        for _ in tqdm(range(self.max_iteration)):
            epoch = self.get_episode()
            G = 0
            for state,action,reward,_ in reversed(epoch):
                self.move_action_defined(action)
                G = self.gamma * G + reward
                # update return value
                # note that we use a modified formula such that we avoid a float overflow
                self.returns[state][0] = self.returns[state][0] * (self.returns[state][1]/(self.returns[state][1] + 1)) + G/(self.returns[state][1] + 1)
                self.returns[state][1] += 1
                self.value[state] = self.returns[state][0]
                
class TD_0(Agent):
    def __init__(self, grid, gamma=0.9, alpha=0.3):
        super().__init__(grid, gamma)
        print(f"This is agent {self.__class__.__name__}")
        self.alpha = alpha
        self.value = [0 for i in range(self.grid.get_size())]
        self.policy = [self.grid.action[random.randint(0,3)]\
                       for i in range(self.grid.get_size())]
    def get_optimal_action(self):
        values = []
        for action in self.grid.action:
            values.append(self.value[self.grid.get_next_state(action)])
        return self.grid.action[values.index(max(values))]       
    def train(self):
        for _ in tqdm(range(self.max_iteration)):
            epoch = self.get_episode()
            for state,_,r,s_prime in epoch:
                self.value[state] = self.value[state] +\
                    self.alpha * (r + self.gamma * self.value[s_prime] - self.value[state])

class OptimalAgent(TD_0):
    # for experiment purpose
    pass

class Experiment():
    def __init__(self, grid, agent, override=True, load_agent=False):
        self.grid = grid
        self.agent = agent
        if load_agent:
            pickle_file = os.path.join(os.path.dirname(__file__),f"{self.agent.__class__.__name__}.pkl")
            if os.path.exists(pickle_file) and override == False:
                with open(pickle_file, 'rb') as file:
                    self.agent = pickle.load(file)
                    self.agent.update_grid(self.grid)
            else:
                self.agent.train()
                with open(pickle_file, 'wb') as file:
                    pickle.dump(self.agent,file) 
        print("Experiment is created")
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

if __name__ == "__main__":
    grid = Gridworld(start=10)
    optimalagent = OptimalAgent(grid=grid)
    exp = Experiment(grid, optimalagent, False)
    exp.train(1000)
    
    agents = [FirstVisitMonteCarlo(grid=grid),EveryVisitMonteCarlo(grid=grid),TD_0(grid=grid),optimalagent]
    results = defaultdict(list)
    for i in range(100):
        for agent in agents:
            exp.set_agent(agent)
            results[agent.__class__.__name__].append(exp.run())
            exp.train(1)
        grid.random_start()
    print(results)

    df = pd.DataFrame(results)
    df = df.sub(df["OptimalAgent"],axis=0)
    df.plot(kind='line',title="Difference from optimal",
            figsize=(10,5),
            xlabel="iteration",ylabel="Difference from optimal",
            grid=True)
    plt.legend(loc='upper right')
    plt.show()
        
    
    