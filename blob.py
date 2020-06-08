# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 11:47:18 2020

@author: ryanc
"""


import numpy as np 
from PIL import Image  
import cv2  # for showing visual live
import matplotlib.pyplot as plt  
import pickle 
from matplotlib import style  
import time 
import os 

os.chdir('C:/Users/ryanc/Documents/boof zone')

style.use("ggplot")  # setting our style!

size = 20 ##set board size

HM_EPISODES = 150000
MOVE_PENALTY = 1
ENEMY_PENALTY = 300
FOOD_REWARD = 25
epsilon = 0
EPS_DECAY = 0.9998  # Every episode will be epsilon*EPS_DECAY
SHOW_EVERY = 1  # how often to play through env visually.

start_q_table = 'qtable-1591640127.pickle' #None or filename of qtable
LEARNING_RATE = 0.1
DISCOUNT = 0.95

PLAYER_N = 1  # player key in dict
FOOD_N = 2  # food key in dict
ENEMY_N = 3  # enemy key in dict

# the dict! Using just for colors
d = {1: (255, 175, 0),  # blueish color
     2: (0, 255, 0),  # green
     3: (0, 0, 255)}  # red

class Blob:
    ##randomly initialize blobs
    def __init__(self):
        self.x = np.random.randint(0,size)
        self.y = np.random.randint(0,size)
            
    ##calculate distance 
    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)
    
    ##possible actions 
    ##four possibilities, potentially add more?
    def action(self, choice):
        if choice == 0:
            self.move(x=1,y=1)
        if choice == 1:
            self.move(x=-1,y=-1)
        if choice == 2: 
            self.move(x=-1,y=1)
        if choice == 3:
            self.move(x=1,y=-1)
        
    ##string method
    def __str__(self):
        return f"{self.x}, {self.y}"
        
    ##move method
    def move(self, x = False, y = False):
        if not x:
            self.x += np.random.randint(-1,2)
        else:
            self.x += x
            
        if not y:
            self.y += np.random.randint(-1,2)
        else:
            self.y += y
        
        ##fix if blobs out of bounds
        
        if self.x < 0:
            self.x = 0
        elif self.x > size - 1:
            self.x = size - 1
            
        if self.y < 0:
            self.y = 0
        elif self.y > size - 1:
            self.y = size - 1
            
## Create q_table
            
if start_q_table is None:
    q_table = {}
    
    for i in range(1-size, size):
        for j in range(1-size, size):
            for k in range(1-size, size):
                for l in range(1-size, size):
                    q_table[((i,j),(k,l))] = [np.random.uniform(-5,0) for i in range(4)]
                    
else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)
        
episode_rewards = []

for episode in range(HM_EPISODES):
    player = Blob()
    food = Blob()
    enemy = Blob()
    if episode % SHOW_EVERY == 0:
        print(f"on #{episode}, epsilon is {epsilon}")
        print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False
    
    episode_reward= 0
    
    for i in range(200):
        ##move player
        obs = (player-food, player - enemy)
        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0,4)
        
        player.action(action)
        
        ##food/enemy motion
        enemy.move()
        food.move()
        
        ##rewarding
        
        if player.x == enemy.x and player.y == enemy.y:
            reward = -ENEMY_PENALTY
        
        elif player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD
        
        else:
            reward = -MOVE_PENALTY
            
        new_obs = (player-food, player-enemy) ##newest observations
        max_future_q = np.max(q_table[new_obs]) 
        current_q = q_table[obs][action]
        
        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        
        else:
            new_q=(1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        
        q_table[obs][action] = new_q
        
        ##visualizig game board
        
        if show:
            env = np.zeros((size, size, 3), dtype=np.uint8)  # starts an rbg of our size
            env[food.x][food.y] = d[FOOD_N]  # sets the food location tile to green color
            env[player.x][player.y] = d[PLAYER_N]  # sets the player tile to blue
            env[enemy.x][enemy.y] = d[ENEMY_N]  # sets the enemy location to red
            img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
            img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
            cv2.imshow("image", np.array(img))  # show it!
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:  # crummy code to hang at the end if we reach abrupt end for good reasons or not.
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        episode_reward += reward
        
        ##break if hit enemy or food
        
        if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
            break
        
    ##track reward in time
    episode_rewards.append(episode_reward)
        
    ##decrease randomness with every episode
    epsilon *= EPS_DECAY
        
##graph results
moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()

with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)