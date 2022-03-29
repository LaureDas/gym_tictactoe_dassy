import numpy as np
import random
import collections
from collections import defaultdict

class QLearningAgent():
    def agent_init(self, agent_init_info):
        """Setup for the agent called when the experiment first starts.
        """
        
        self.num_actions = agent_init_info["num_actions"]
        self.epsilon = agent_init_info["epsilon"]
        self.gamma = agent_init_info["gamma"]
        self.alpha = agent_init_info["alpha"]
        self.rand_generator = np.random.RandomState(agent_init_info["seed"])
        self.q = defaultdict(float)

    def agent_start(self, state):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            observation (Numpy array): the state observation from the environment's env_start function.
        Returns:
            The first action the agent takes.
        """
     # Choose action using epsilon greedy.
        current_q=self.q.setdefault(state,[0,0])
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action= self.argmax(current_q)
        self.prev_action=action
        self.prev_state=state
        # print('agent q', self.q)
        return action

    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (int): the state from the
                environment's step based on where the agent ended up after the
                last step.
        Returns:
            action (int): the action the agent is taking.
        """    
        # Choose action using epsilon greedy.
        current_q = self.q.setdefault(state,[0,0])
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)

        target=reward+self.gamma*np.max(self.q[state])
        # print('testing',self.q[self.prev_state,self.prev_action])
        update=self.alpha*(target-self.q[self.prev_state,self.prev_action])
        self.q[self.prev_state,self.prev_action]+= update

        # print('update', update)
        # print('list',self.q)
        # print('id problem',self.q[self.prev_state][self.prev_action])
        self.prev_state = state
        self.prev_action = action
        return action
    
    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """    
        target = reward #for terminal state target = reward only, as no look ahead state exist
        update=self.alpha*(target-self.q[self.prev_state, self.prev_action])
        # update=self.q[self.prev_state]
        # update[self.prev_action] += self.alpha*(target - self.q[self.prev_state, self.prev_action])
        self.q[self.prev_state, self.prev_action] += update
        # self.q[self.prev_state, self.prev_action] = self.q[self.prev_state, self.prev_action] + self.step_size * (reward + 0 - self.q[self.prev_state, self.prev_action])

    def argmax(self, q_values):
        """argmax with random tie-breaking
        Args:
            q_values (Numpy array): the array of action-values
        Returns:
            action (int): an action with the highest value
        """
        top = float("-inf")
        ties = []

        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []

            if q_values[i] == top:
                ties.append(i)

        return self.rand_generator.choice(ties)

    def set_epsilon(self, value):
        self.epsilon=value