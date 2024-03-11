from abc import ABC
import random


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
    def get_training_state(self):
        unvisited = [i for i in range(self.size)]
        for i in self.goal:
            unvisited.remove(i)
        # unvisited.remove(self.start)
        # unvisited.insert(0,self.start)
        return unvisited
    def update_state(self, action):
        self.state = self.get_next_state(action)

class Agent(ABC):
    def __init__(self, grid, gamma, max_iteration=1000000):
        self.grid = grid
        self.gamma = gamma
        self.reward = 0
        self.done = False
        self.reward = 0
        self.max_iteration = max_iteration
        print("Agent is created")

    def get_optimal_action(self):
        pass
    
    def train(self):
        pass
    
    def move(self):
        action = self.get_optimal_action()
        self.grid.update_state(action)
        return action

    def reset(self):
        self.state = self.grid.start
        self.reward = 0
    
    def start(self):
        print(f"The starting point is {self.grid.start}")
        print(f"This {self.__class__.__name__} will start now.")
        self.grid.reset()
        action = []
        policy = None
        while not self.grid.is_finish():
            print(f"state was {self.grid.get_state()}", end=', ')
            policy = self.move()
            print(f"action is {policy}, state is now {self.grid.get_state()}")
            action.append(policy)
        print(f"Number of steps taken is {len(action)}")
        print(f"The action taken are {action}")
        print(f"==================================================")

class RandomAgent(Agent):
    def get_optimal_action(self):
        return self.grid.action[random.randint(0,3)]
    
class IterativePolicyEvaluation(Agent):
    def __init__(self, grid, gamma, max_iteration, theta=0.1):
        super().__init__(grid, gamma, max_iteration)
        print(f"This is agent {self.__class__.__name__}")
        self.theta = theta
        self.value = [0 for i in range(self.grid.get_size())]
        self.policy = [self.grid.action[random.randint(0,3)]\
                       for i in range(self.grid.get_size())]
        
    def get_optimal_action(self):
        return self.policy[self.grid.get_state()]
    
    def get_value(self,state):
        return self.value(state)
    
    def iterate_values(self):
        values = []
        for action in self.grid.action:
            values.append(self.grid.action_prob[action]*
                          (self.grid.get_reward() + 
                           self.gamma *  self.value[self.grid.get_next_state(action)]))
        return values
    def policy_evaluation(self):
        delta = self.theta+1
        while delta > self.theta:
            # Evaluate
            delta = 0
            self.grid.reset()
            for i in self.grid.get_training_state():
                self.grid.state = i
                v = self.value[i]
                self.value[i] = sum(self.iterate_values())
                delta = max(abs(v-self.value[i]),delta)
    def policy_improvement(self):
        stable = True
        for i in self.grid.get_training_state():
            # Improve
            self.grid.state = i
            old_policy = self.get_optimal_action()
            values = self.iterate_values()
            new_policy = self.grid.action[values.index(max(values))]
            self.policy[self.grid.get_state()] = new_policy
            if old_policy != new_policy:
                stable = False
        return stable
    def train(self):
        stable = False
        count = 0
        while not stable and count<self.max_iteration:
            self.policy_evaluation()
            stable = self.policy_improvement()
            count+=1
        
class ValueIteration(Agent):
    def __init__(self, grid, gamma, theta=0.1):
        super().__init__(grid, gamma)
        print(f"This is agent {self.__class__.__name__}")
        self.theta = theta
        self.value = [0 for i in range(self.grid.get_size())]
    def get_optimal_action(self):
        values = self.iterate_values()
        return self.grid.action[values.index(max(values))]     
    def iterate_values(self):
        values = []
        for action in self.grid.action:
            values.append(self.grid.action_prob[action]*
                          (self.grid.get_reward() + 
                           self.gamma * self.value[self.grid.get_next_state(action)]))
        return values
    def train(self):
        delta = self.theta+1
        while delta > self.theta:
            delta = 0
            self.grid.reset()
            for i in self.grid.get_training_state():
                self.grid.state = i
                v = self.value[i]
                self.value[i] = max(self.iterate_values())
                delta = max(abs(v-self.value[i]),delta)
    

if __name__ == "__main__":
    grid_success = False
    while not grid_success:
        try:  
            grid = Gridworld(start=30)
            grid_success = True
        except:
            print("Start cannot be the goal (1,35). Try again.") 
    
    agent = RandomAgent(grid, 0.9)
    agent.start()
    agent = IterativePolicyEvaluation(grid=grid, gamma=0.9,
                                      max_iteration=1000,theta=0.00001)
    agent.train()
    agent.start()
    agent = ValueIteration(grid=grid, gamma=0.99, theta=0.00001)
    agent.train()
    agent.start()