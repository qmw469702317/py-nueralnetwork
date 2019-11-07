#!/usr/bin/env python
# coding: utf-8

# In[31]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all" 


# In[18]:


#import libs
import numpy as np
import matplotlib.pylab as plt
import scipy.special as sc


# In[64]:


#active functions

def step_function(x):
    return np.array(x>0,dtype=np.int)

def active_function(x):
    return sc.expit(x)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0,x)

#一般用于回归问题，即根据某个输入预测一个连续的值的问题
def identity_function(a):
    return a

#一般用于分类问题，即男性还是女性这种,实际应用中因为指数
#运算需要一定运算量，常被省略
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)     #减去最大常数c避免溢出风险
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y




# In[62]:


def init_network():
    network={}
    network['W1'] = np.array([[0.5,0.3,0.1],[0.2,0.4,0.7]])
    network['b1'] = np.array([0.2,0.1,0.4])
    network['W2'] = np.array([[0.4,0.1],[0.2,0.8],[0.3,0.4]])
    network['b2'] = np.array([0.2,0.4])
    network['W3'] = np.array([[0.4,0.3],[0.5,0.9]])
    network['b3'] = np.array([0.1,0.2])
    
    return network

def forward(network,x):
    W1,W2,W3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']
    
    a1 = np.dot(x,W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3) + b3
    y = identity_function(a3)
    
    return y


# In[63]:


network = init_network()
x = np.array([1.0,0.5])
y = forward(network,x)
print(y)

y = 

