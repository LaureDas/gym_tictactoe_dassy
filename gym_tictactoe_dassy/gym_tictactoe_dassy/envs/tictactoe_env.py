import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import random

class TictactoeEnv(gym.Env):
    """
    Implementation of a TicTacToe Environment based on OpenAI Gym standards
    """
    #------------------------------------GAME INIT-------------------------------------------
    def __init__(self):
        self.observation_space = spaces.Box(low=np.array([-1]*9), high=np.array([1]*9), dtype=np.int)
        #board as 3x3, 9 spaces required
        self.action_space = spaces.Discrete(9)

        #player1 1=cross, player2: -1 =knot; nothing yet=0
        self.symbols = { 0:' ', -1:'O', 1:'X'}
        #these are the available spots on my board 
        #BOARD
        #[0,1,2]
        #[3,4,5]
        #[6,7,8]
        #status is 2 for play, 1 for win, -1 for lose and 0 for draw, 
        self.status = 2
        #all available positions on the board 
        self.available = [0,1,2,3,4,5,6,7,8]
        #obervsation space =state space defining the set of all states of the environments you can be in 
    #------------------------------------GAME STATE------------------------------------------


    #if remaining in row of spot  = symbol return true
    def check_horiztontal(self,spot,symbol):
        rows = [[0,1,2],[3,4,5],[6,7,8]]
        for row in rows:
            if spot in row:
                row.remove(spot)
                for p in row:
                    if self.current_state[p]!=symbol:
                        return False
                return True
 
    #if remaining in col of spot  = symbol return true        
    def check_vertical(self,spot,symbol):
        columnts = [[0,3,6],[1,4,7],[2,5,8]]
        for c in columnts:
            if spot in c:
                c.remove(spot)
                for p in c:
                    if self.current_state[p]!=symbol:
                        return False
                return True
    
    #if remaining in diagonals of spot  = symbol return true 
    def check_diagonal(self,spot,symbol):
        diagonals = [[0,4,8],[2,4,6]]
        if spot != 4:
            for d in diagonals:
                if spot in d:
                    d.remove(spot)
                    for p in d:
                        if self.current_state[p]!=symbol:
                            return False
                    return True
        else:
            if (self.current_state[2]==symbol and self.current_state[6]==symbol) or (self.current_state[0]==symbol and self.current_state[8]==symbol):
                return True
            else:
                return False
    
    def is_draw(self,spot,symbol):
        pieces_left=len(self.available)
        #if there's only one piece avail and no one had won then, its a draw
        if pieces_left==1:
            if not self.is_win(spot,symbol):
                return True
        else:
            return False

    #check through all states defined above if one of them is true --> then we have a winner
    def is_win(self,spot,symbol):
        if self.check_horiztontal(spot,symbol):
            return True
        if self.check_diagonal(spot,symbol):
            return True
        return self.check_vertical(spot,symbol)
     
    #check through all states defined above if one of them is -1 (=lose)
    def is_block(self,spot,symbol):
        neg_symb=-1*symbol
        if self.check_horiztontal(spot,neg_symb):
            return True
        if self.check_vertical(spot,neg_symb):
            return True
        return self.check_diagonal(spot,neg_symb)

    #------------------------------------ACTION------------------------------------------
    #step method
    def step(self,action):
        """
            Given an action it returns
            * an observations
            * a reward
            * a status report if the game is over
            * an info dictionary
        """
        completed = False
        #agent turn then other player turn
        #if not there then, we're done with the game
        if action not in self.available:
            completed = True
            reward = -5
            self.status = -1
        else:
            if self.is_win(action,1):
                self.status = 1
                self.current_state[action] = 1
                self.available.remove(action)
                reward = 2
                completed = True
            elif self.is_draw(action,1):
                self.status = 0
                self.current_state[action] = 1
                self.available.remove(action)
                reward = 1
                completed = True
            elif self.is_block(action,1):
                self.status = 2
                self.current_state[action] = 1
                self.available.remove(action)
                reward = 1
            else:
                self.status = 2
                self.current_state[action] = 1
                self.available.remove(action)
                reward = 0
            
            #other player turn
            if self.status == 2:
                self.other_player_turn()
                if self.status == -1:
                    reward +=-2 
                    completed = True
                elif self.status == 0:
                    reward += 1
                    completed = True
                elif self.status == 1:
                    completed = True #no reward as other player lost
        return tuple(self.current_state),reward,completed,{'available_spot':self.available, 'game_status':self.status}
    
    def other_player_turn(self):
        #random choice for the spot the other player will pick
        action = random.sample(self.available,1)[0]
        if self.is_win(action,-1):
            self.status = -1
        elif self.is_draw(action,-1):
            self.status = 0
        self.current_state[action] = -1
        self.available.remove(action)

    #------------------------------------DISPLAY------------------------------------------
    def reset(self):
        #reset init parameters
        self.available = [0,1,2,3,4,5,6,7,8]
        self.status = 2
        
        #decide who starts randomly
        random_toss=random.random()
        if random_toss<0.5:
            #player1 turn
            self.current_state = [0]*9
        else:
            #player2 turn
            action = random.sample(self.available,1)[0]
            state = [0]*9
            state[action] = -1
            self.current_state = state
            self.available.remove(action)
    
    #PRINT BOARD
    def display_board(self): 
        grid = '{}|{}|{}'
        print(grid.format(self.symbols[self.current_state[0]] , self.symbols[self.current_state[1]], self.symbols[self.current_state[2]]))
        print('-----')
        print(grid.format(self.symbols[self.current_state[3]], self.symbols[self.current_state[4]], self.symbols[self.current_state[5]]))
        print('-----')
        print(grid.format(self.symbols[self.current_state[6]], self.symbols[self.current_state[7]],self.symbols[self.current_state[8]]))
        
    def render(self, mode='console'):
        if mode!='console':
            raise NotImplementedError()
        else:
            self.display_board()
        
    def close(self):
        pass