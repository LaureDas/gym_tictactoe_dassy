import numpy as np
import random
import collections

def getPolicy(Q):
    policy=collections.defaultdict(lambda : ValueError('Not Defined'))
    for s in Q.keys():
        policy[s] = np.argmax(Q[s])
    
    def greedyAction(s):
        action = policy[s]
        return action
    return greedyAction

def qLearning(Q,env,gamma, alpha, epsilon, episodes, opponent_policy = None):
    total_rewards = []
    for e in range(episodes):
        #start state, assuming enviroment return integer as observation
        s =env.reset()
        #set opponent type for every game
        
        num_actions = len(Q[s])
        if np.random.rand() < epsilon:
            a = np.argmax(Q[s])
        else:
            a = np.random.randint(0, num_actions)

        done = False
        total_reward = 0
        while not done: #and i< max_steps:
            s2, r, done,info = env.step(a) 
            total_reward += r
            if done:
                target = r #for terminal state target = reward only, as no look ahead state exist
                update = alpha*(target - Q[s][a])
                Q[s][a] += update
            else:
                a2 = np.argmax(Q[s2])
                target = r + gamma*Q[s2][a2]
                update = alpha*(target - Q[s][a])
                Q[s][a] += update
                s = s2
                                
                num_actions2 = len(Q[s2])
                if np.random.rand() < epsilon:
                    a = np.argmax(Q[s2])
                else:
                    a = np.random.randint(0, num_actions2)
        total_rewards.append(total_reward)
    policy =  getPolicy(Q)      
    return policy, total_rewards       