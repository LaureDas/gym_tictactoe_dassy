
#import my libraries
import gym
import gym_tictactoe_dassy
from QLearning import qLearning
from Qlearn_agent import QLearningAgent
import pickle as pkl
from tqdm import tqdm

#import common libraries
import argparse
import random
import numpy as np
np.random.seed(0)
random.seed(0)
import collections
from matplotlib import pyplot as plt

def play(env,policy, episodes):
    print('Playing..') 
    test_rewards = []
    result = {'win':0,'draw':0,'lose':0}
    for e in range(episodes):
        print('\nNew episode {} starting..\n'.format(e))
        state = env.reset()
        done = False
        env.render()
        total_reward = 0
        while not done: 
            action = policy(state)
            print('\nmarking X at spot:{}'.format(action))
            state, r, done, info = env.step(action)
            total_reward += r
            if done:
                print('\nEpisode finished') 
                status = info['game_status']
                print('\nGame Status: {}'.format(status))
                if status == 1:
                    result['win']+=1
                elif status == -1:
                    result['lose']+=1
                elif status == 0:
                    result['draw']+=1
            else:
                env.render()
                print('\nGame Status: {}'.format(info['game_status']))
                print('\nAvailable Positions:{}'.format(info['available_spot']))
        test_rewards.append(total_reward)
    return result, test_rewards


def moving_average(x, size = 1000):
    x = np.array(x)
    means = []
    for i in range(size,len(x), size):
        mean = np.mean(x[max(0,i-size):i])
        means.append(mean)
    return means
    
def getPolicy(Q):
    policy=collections.defaultdict(lambda : ValueError('Not Defined'))
    for state in Q.keys():
        policy[state] = np.argmax(Q[state])
    
    def greedyAction(state):
        action = policy[state]
        return action
    return greedyAction

if __name__=="__main__": 
    np.random.seed(0)
    parser = argparse.ArgumentParser()
    #indicate where to save results 
    args = parser.parse_args()
    
    #environment
    env = gym.envs.make('tictactoe-v0')
    #training parameters
    num_actions=9
    gamma = 0.99
    alpha = 0.4
    epsilon = 0.1
    train_episodes = 10000
    ep_decay=0.9

    #agent initialization
    agent_info = {"num_actions": num_actions, "epsilon": epsilon, "gamma": gamma, "alpha": alpha, "seed":0}
    agent = QLearningAgent()
    agent.agent_init(agent_info)

    train_rewards = [] 
    progress = []
    for eps in [200]*int(train_episodes/200):
        #training
        epsilon_value=0.1
        print("Start q-leaner training ")
        dic_rewards = {}
        policy={}
        for step in tqdm(range(train_episodes), position=0):  
            current_reward=0
            #start state, assuming enviroment return integer as observation
            s=env.reset()
            #set opponent type for every game
            action=agent.agent_start(s)
            if step % 5000 == 0:
                #runs episode
                epsilon = epsilon_value*ep_decay
                agent.set_epsilon(epsilon)
            while True:
                s2, rwd, completed, info=env.step(action)
                # print('s2',s2)
                # print('rwd',rwd)
                # print('complete', completed)
                current_reward+= rwd
                if completed:
                    agent.agent_end(rwd)
                    dic_rewards[step]=(current_reward)                  
                    break
                else:
                    # print('there')
                    action=agent.agent_step(rwd,s2)
    
        # print('there', list(dic_rewards.values()))
        pkl.dump(agent,open("qlearner.pkl", "wb"))
        policy=getPolicy(agent.q)
        train_rewards += dic_rewards
        result, _ = play(env,policy,100)
        progress.append(result['win'])
 
  # # Progress during traininig
    #save plots images
    plt.plot(range(200,train_episodes+1,200),progress)
    plt.xlabel('Episodes')
    plt.ylabel('Number of wins(out of 100 test games)')
    plt.title('Progress(after every 200 train games against rand player)')
    plt.savefig('training_progress_against_ONLY_rand_opponent.png')


#FIX THIS 
    size = 10000
    rewards_list = list(dic_rewards.values())
    mmeans = moving_average(rewards_list, size = size)
    plt.plot([size*episode for episode in range(0,len(mmeans))],mmeans, label=agent)
    plt.xlabel("Episodes")
    plt.ylabel("Rewards Total", labelpad=35)
    plt.legend()
    plt.show()
