#!/usr/bin/env python
# coding: utf-8

# In[542]:


import numpy
import scipy.special
import matplotlib.pyplot
get_ipython().run_line_magic('matplotlib', 'inline')

# neural netwoek class definition
class neuralNetwork:
  
    #initialise the neural network
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        #set number of nodes in each input,hidden,output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        #learning rate
        self.lr = learningrate
        
        #weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        #w11 w21
        #w12 w22 etc
        self.wih = numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who = numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        
        self.activation_function = lambda x:scipy.special.expit(x)
        pass
    
    #train the neural network
    def train(self,inputs_list,targets_list):
        inputs = numpy.array(inputs_list,ndmin=2).T
        targets = numpy.array(targets_list,ndmin=2).T
        
        hidden_inputs = numpy.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T,output_errors)
        
        self.who += self.lr*numpy.dot((output_errors*final_outputs*(1.0 - final_outputs)),numpy.transpose(hidden_outputs))
        
        self.wih += self.lr*numpy.dot((hidden_errors*hidden_outputs*(1.0 - hidden_outputs)),numpy.transpose(inputs))
        
        
        pass
    
    #query the neural network
    def query(self,inputs_list):
        inputs = numpy.array(inputs_list,ndmin=2).T
        
        hidden_inputs = numpy.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
        pass


# In[543]:


input_nodes = 784
hidden_nodes = 100
output_nodes = 10

learning_rate = 0.1

n = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)


# In[544]:


train_data_file = open("C:\\Users\\1\\Documents\\mnist_train_100.csv",'r')
train_data_list = train_data_file.readlines()
train_data_file.close()


# In[545]:


for record in train_data_list:
    all_values = record.split(',')
    inputs = (numpy.asfarray(all_values[1:])/255*0.99) + 0.01
    targets = numpy.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs,targets)
    pass


# In[546]:


test_data_file = open("C:\\Users\\1\\Documents\\mnist_test_10.csv",'r')
test_data_list = test_data_file.readlines()
test_data_file.close()


# In[547]:


all_values = test_data_list[0].split(',')
print(all_values[0])
image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
matplotlib.pyplot.imshow(image_array,cmap='Greys',interpolation='None')


# In[548]:


n.query((numpy.asfarray(all_values[1:])/255.0*0.99)+0.01)


# In[ ]:





# In[ ]:




