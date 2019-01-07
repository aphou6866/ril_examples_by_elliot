import os, sys, json
import numpy as np 
import tensorflow as tf
import gym
from ql_method import dqn_method
from ql_networks import build_dense_network, build_dense_duel


np.random.seed(1)
tf.set_random_seed(1)

env = gym.make('CartPole-v0')
env = env.unwrapped
MEMORY_SIZE = 1000

# python3 dqn_cartpole.py 3000


if __name__ == "__main__":
    
    argv=sys.argv
    itr= int(argv[1])
    
    sess = tf.Session()
    dqn= dqn_method( sess, build_dense_network, n_actions=env.action_space.n, n_features=env.observation_space.shape[0], layers=1, hiddens=20, 
                    memory_size=MEMORY_SIZE, e_greedy_increment=0.001)
    dqn.set_replacement( None, 300, 501)
    sess.run(tf.global_variables_initializer())
    
    steps=0
    ep_r= 0
    for i in range(itr):
        done=False
        observation = env.reset()
        
        while done==False:
            env.render()
            action= dqn.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            
            x, x_dot, theta, theta_dot = observation_
            r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
            reward = r1 + r2
            
            dqn.store_transition(observation, action, reward, observation_)
            ep_r= 0.9*ep_r + 0.1*reward
            
            if steps%100==0:
                print('%d, %5d Reward:%f'%(i, steps, ep_r))
            if steps > MEMORY_SIZE or 0< i:   # learning
                    dqn.learn()
            
            observation = observation_
            steps += 1
