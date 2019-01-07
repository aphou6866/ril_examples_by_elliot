import os, sys, json
import numpy as np 
import tensorflow as tf
import gym, time
import multiprocessing
import threading
from threading import Lock, Thread
from ql_method import dqn_method
from ql_networks import build_dense_network, build_dense_duel


np.random.seed(1)
tf.set_random_seed(1)
MEMORY_SIZE = 1000
layers=1
hiddens=20

lock = Lock()
tol_itr=0

def worker( pnum, COORD, env, dqn, steps, itr):
    global tol_itr
    
    print("Runnint worker-%d"%pnum)
    for i in range(itr):
        if COORD.should_stop()==True:
            break
        
        while (tol_itr// N_WORKERS) !=i:
            time.sleep(0.1)
            
        observation = env.reset()
        ep_r= 0
        for j in range(steps):
            
            if pnum==0 and 10<i:
                env.render()
            action= dqn.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            if done:
                reward = 2
            dqn.store_transition(observation, action, reward, observation_)
            
            ep_r= 0.9*ep_r + 0.1*reward
            
            if pnum==0 and j%500==0:
                print('p%d %d, %5d Reward:%f'%(pnum, i, j, ep_r))
            
            if j > MEMORY_SIZE :   # learning
                dqn.learn()
                
            stp= j
            if done:
                break
            observation = observation_
        dqn.send_network()
        
        lock.acquire()
        tol_itr +=1
        lock.release()
        
        print('p%d %d( %d) total steps:%d reward:%f'%(pnum, i, tol_itr, stp, ep_r)) 

def a2dqn( steps, itr):
    global N_WORKERS
    N_WORKERS = multiprocessing.cpu_count()
    
    sess = tf.Session()
    #dqn_parent= dqn_method( build_dense_network, n_actions=env.action_space.n, n_features=env.observation_space.shape[0], layers=1, hiddens=20, 
                    #memory_size=MEMORY_SIZE, replace_target_iter=300, e_greedy_increment=0.001)
    e = gym.make('MountainCar-v0')
    e = e.unwrapped
    _, _, parent=  build_dense_network( e.observation_space.shape[0], e.action_space.n , layers, hiddens, 'parent' )
    
    env= []
    dqn= []
    for i in range(N_WORKERS):
        e = gym.make('MountainCar-v0')
        e = e.unwrapped
        env.append( e)
        d=  dqn_method( sess, build_dense_network, n_actions=e.action_space.n, n_features=e.observation_space.shape[0], layers=layers, hiddens=hiddens, 
                        memory_size=MEMORY_SIZE, e_greedy_increment=0.001)
        dqn.append( d)
    for i in range(N_WORKERS):
        if i==0:
            dqn[i].set_replacement( dqn[ N_WORKERS-1].n_params, 100, 200)
        else:
            dqn[i].set_replacement( dqn[ i-1].n_params, 100, 200)
            
    COORD = tf.train.Coordinator()
    sess.run(tf.global_variables_initializer())
    
    worker_threads = []
    for i in range(N_WORKERS):
        t = threading.Thread(target=worker, args=( i, COORD, env[i], dqn[i], steps, itr ) )
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)
    
    
def dqn( steps, itr):
    global N_WORKERS
    N_WORKERS= 1
    
    env= gym.make('MountainCar-v0')
    env= env.unwrapped
    
    sess = tf.Session()
    dqn= dqn_method( sess, build_dense_network, n_actions=env.action_space.n, n_features=env.observation_space.shape[0], layers=layers, hiddens=hiddens, 
                        memory_size=MEMORY_SIZE, e_greedy_increment=0.001)
    dqn.set_replacement( None, 500, 501)
    
    COORD = tf.train.Coordinator()
    sess.run(tf.global_variables_initializer())
    
    worker( 0, COORD, env, dqn, steps, itr)
    

# python3 dqn_mountaincar.py  3000 1000 dqn


if __name__ == "__main__":
    
    argv=sys.argv
    steps= int(argv[1])
    itr= int(argv[2])
    
    if argv[3]=='a2dqn':
        a2dqn( steps, itr)
    else:
        dqn( steps, itr)
    
    

#if __name__ == "__main__":
    
    #argv=sys.argv
    #steps= int(argv[1])
    #itr= int(argv[2])
    
    #dqn= dqn_method( build_dense_network, n_actions=env.action_space.n, n_features=env.observation_space.shape[0], layers=1, hiddens=20, 
                    #memory_size=MEMORY_SIZE, replace_target_iter=300, e_greedy_increment=0.001)
    
    #for i in range(itr):
        
        #observation = env.reset()
        #ep_r= 0
        #for j in range(steps):
            
            #env.render()
            #action= dqn.choose_action(observation)
            #observation_, reward, done, info = env.step(action)
            #dqn.store_transition(observation, action, reward, observation_)
            
            #ep_r= 0.9*ep_r + 0.1*reward
            
            #if j%100==0:
                #print('%d, %5d Reward:%f'%(i, j, ep_r))
            
            #if j > MEMORY_SIZE :   # learning
                #dqn.learn()
                
            #if done:
                #break
            #observation = observation_
            
            
