
"""
The DQN improvement: Prioritized Experience Replay (based on https://arxiv.org/abs/1511.05952)
From Morvan Python website https://morvanzhou.github.io/tutorials/

Modified by Elliot Hou, Ao-ping, www.aiacademy.tw

Using:
Tensorflow: 1.4
gym: 0.8.0
"""

import os, sys, json
import numpy as np 
import tensorflow as tf


class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Story data with its priority in the tree.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)
            
            

class dqn_method(object):
    
    def __init__(
            self,
            sess,
            build_network,
            n_actions,
            n_features,
            layers= 1,
            hiddens= 20,
            learning_rate=0.005,
            reward_decay=0.9,
            e_greedy=0.9,
            memory_size=10000,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
            prioritized=True,
            double_q=False,
            pnum=0, 
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.prioritized = prioritized    # decide to use double q or not
        self.double_q = double_q    # decide to use double q or not
        self.learn_step_counter = 0
        self.pnum= pnum
        
        self.n_X, self.n_Y, self.n_params= build_network( n_features, n_actions, layers, hiddens, 'next%d'%pnum)
        self.e_X, self.e_Y, self.e_params= build_network( n_features, n_actions, layers, hiddens, 'eval%d'%pnum)
        self.build_method( self.e_Y, prioritized, learning_rate)
        
        #if self.parent==None:
            #self.replace_target_op = [tf.assign( n, e) for n, e in zip(n_params, e_params)]
        #else:
            #self.replace_target_op = [tf.assign( n, e) for n, e in zip(n_params, parent)]
            #self.update_target_op = [tf.assign( n, e) for n, e in zip(parent, e_params)]
        
        if self.prioritized:
            self.memory = Memory(capacity=memory_size)
        else:
            self.memory = np.zeros((self.memory_size, n_features*2+2))

        self.sess = sess
        #if sess is None:
            #self.sess = tf.Session()
            #self.sess.run(tf.global_variables_initializer())
        #else:
            #self.sess = sess
        #if output_graph:
            #tf.summary.FileWriter("logs/", self.sess.graph)
        
        
    def build_method( self, Y, prioritized, lr):
        
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target') # for calculating loss
        if prioritized==True:
            self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')
            self.abs_errors = tf.reduce_sum(tf.abs(self.q_target - self.e_Y), axis=1)    # for updating Sumtree
            self.loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.q_target, self.e_Y))
        else:
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.e_Y))
        
        self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
        
    
    def set_replacement( self, next_params, sync_itr, replace_itr):
        assert sync_itr < replace_itr, 'replace_itr(%d) have to larger then sync_itr(%d)'%(sync_itr, replace_itr)
        self.replace_itr= replace_itr
        self.sync_itr= sync_itr
        
        self.sync_op= [tf.assign( n, e) for n, e in zip( self.n_params, self.e_params)]
        if next_params!= None:
            self.replace_op= [tf.assign( n, e) for n, e in zip( next_params, self.e_params)]
        else:
            self.replace_op= None
    
    

    def store_transition(self, s, a, r, s_):
        if self.prioritized:    # prioritized replay
            transition = np.hstack((s, [a, r], s_))
            self.memory.store(transition)    # have high priority for newly arrived transition
        else:       # random replay
            if not hasattr(self, 'memory_counter'):
                self.memory_counter = 0
            transition = np.hstack((s, [a, r], s_))
            index = self.memory_counter % self.memory_size
            self.memory[index, :] = transition
            self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.e_Y, feed_dict={self.e_X: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action
    
    def pick_action(self, observation):
        observation = observation[np.newaxis, :]
        actions_value = self.sess.run(self.e_Y, feed_dict={self.e_X: observation})
        action = np.argmax(actions_value)
        #print(observation, action, actions_value)
        return action
    
    
    def send_network( self):
        if self.replace_op!= None:
            self.sess.run(self.replace_op)
        
        
        
    def learn(self):
        if self.learn_step_counter % self.sync_itr == 0:
            self.sess.run(self.sync_op)
            #print('target_params_replaced 1')
        
        #if self.replace_op!= None:
            #if self.learn_step_counter % self.replace_itr == 0:
                #self.sess.run(self.replace_op)
                ##print('target_params_replaced 2')
            
        if self.prioritized:
            tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
            batch_memory = self.memory[sample_index, :]

        
        q_next, q_eval = self.sess.run(
                                [self.n_Y, self.e_Y],
                                feed_dict={self.n_X: batch_memory[:, -self.n_features:],
                                self.e_X: batch_memory[:, :self.n_features]}
                                )
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]
    
        #q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)
        if self.double_q:
            max_act4next = np.argmax(q_eval, axis=1)        # the action that brings the highest value is evaluated by q_eval
            selected_q_next = q_next[batch_index, max_act4next]  # Double DQN, select q_next depending on above actions
        else:
            selected_q_next = np.max(q_next, axis=1)    # the natural DQN

        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next
        
        if self.prioritized:
            _, abs_errors, self.cost = self.sess.run([self._train_op, self.abs_errors, self.loss],
                                         feed_dict={self.e_X: batch_memory[:, :self.n_features],
                                                    self.q_target: q_target,
                                                    self.ISWeights: ISWeights})
            self.memory.batch_update(tree_idx, abs_errors)     # update priority
        else:
            _, self.cost = self.sess.run([self._train_op, self.loss],
                                        feed_dict={self.e_X: batch_memory[:, :self.n_features],
                                        self.q_target: q_target})
        
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        

        
    def load(self, sess, modelName):
        jfn= "./"+modelName+".json"
        if os.path.isfile(jfn):
            with open(jfn) as f:
                info= json.load(f)
                
            if info['pnum']!= self.pnum:
                print("Inconsist pnum: %d(%d)"%(info['pnum'], self.pnum))
                return float('-inf')
            fn= info['filename']
            saver= tf.train.Saver( self.n_params +  self.e_params)
            saver.restore( sess, fn)
            print("Load model:", fn, 'with reward', info['reward'])
            return info['reward']
        return float('-inf')
    

    def save(self, sess, modelName, reward):
        jfn= "./"+modelName+".json"
        fn= "./%s-%d.ckpt"%(modelName, self.pnum)
        
        saver= tf.train.Saver( self.n_params +  self.e_params)
        save_path = saver.save( sess, fn)
        info= {'pnum':self.pnum, 'reward': reward, 'filename':fn}
        
        with open( jfn, "w") as f:
            f.write('%s'%json.dumps(info) )
            f.write('\n')
            
        print('Save model ', modelName, " to path", save_path, 'with reward', info['reward'], 'pnum',self.pnum)
        
        
        
        
    
    
    
    
