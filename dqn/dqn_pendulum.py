import os, sys, json
import numpy as np 
import tensorflow as tf
import gym
from ql_method import dqn_method
from ql_networks import build_dense_network, build_dense_duel

np.random.seed(1)
tf.set_random_seed(1)
env = gym.make('Pendulum-v0')
env = env.unwrapped
env.seed(1)
MEMORY_SIZE = 3000
ACTION_SPACE = 25


def dqn( steps, itr):
    
    sess = tf.Session()
    #dqn= dqn_method( build_dense_network, n_actions=ACTION_SPACE, n_features=3, layers=1, hiddens=30, memory_size=MEMORY_SIZE, replace_target_iter=300, e_greedy_increment=0.001)
    dqn= dqn_method( sess, build_dense_duel, n_actions=ACTION_SPACE, n_features=3, layers=1, hiddens=30, memory_size=MEMORY_SIZE,  e_greedy_increment=0.001)
    dqn.set_replacement( None, 300, 501)
    sess.run(tf.global_variables_initializer())
    ep_r= -300
    for i in range(itr):
        total_steps = 0
        observation = env.reset()
        
        for j in range(steps):
            #if 2<i:
            env.render()
            action= dqn.choose_action(observation)
            
            f_action = (action-(ACTION_SPACE-1)/2)/((ACTION_SPACE-1)/4)   # convert to [-2 ~ 2] float actions
            observation_, reward, done, info = env.step(np.array([f_action]))
            dqn.store_transition(observation, action, reward/10, observation_)
            
    
            if j%100==0:
                print('%d, %5d Reward:%f'%(i, j, ep_r))
            
            if j > MEMORY_SIZE or 0< i:   # learning
                dqn.learn()
                ep_r = ep_r*0.9 + 0.1*reward
            else:
                ep_r= reward
                
            observation = observation_


# python3 dqn_pendulum.py  20000 1000 dqn


if __name__ == "__main__":
    
    argv=sys.argv
    steps= int(argv[1])
    itr= int(argv[2])
    
    if argv[3]== 'dqn':
        dqn( steps, itr)
    
