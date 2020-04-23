#!/usr/bin/env python 
# Author: Ritvik Kapila
# coding: utf-8

# In[1]:


# Importing the necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# Reading the training and testing data

df=pd.read_csv('mnist_train.csv', header=None)
dftest=pd.read_csv('mnist_test.csv', header=None)
dfout=df[0]
dfinp=df.drop([0],axis=1)
x=np.array(dfinp)
print(dfinp)
dftest[784]=1
x_test=np.array(dftest)
x_test
# dfinp=dfinp.append(dfinp.iloc[0])
# dfinp.reset_index(inplace=True)
# dfinp=dfinp.drop(['index'],axis=1)
# dfinp.iloc[7000]=1
# dfinp


# In[4]:


# Visualization of a data point in the form of a 28X28 matrix

def Input_Visualize(X,i):
    plt.imshow(X[i].reshape(28,28))
    
Input_Visualize(x,0)


# In[ ]:


# Adding biases to the neurons

dfinp[785]=1
print(dfinp)
x=np.array(dfinp)
y=np.array(dfout)


# In[ ]:


# Forming the desired values for the corresponding output neurons according to the given labels

def Out_Array(n):
    a=[];
    for i in range(0,10):
        if i==n:
            a=a+[1]
        else:
            a=a+[0]
    a=a+[0]        
    return a        
            
print(Out_Array(8))


# In[ ]:


# Activation Functions used in the network

def activation_function(z,type):
    if(type=='sigmoid'):
        return 1/(1+np.exp(-z))
    if(type=='tanh'):
        return np.tanh(z)
    if(type=='ReLu'):
        return np.maximum(0,z)
    if(type=='softmax'):
        B = np.exp(z)
        B = B/(np.sum(B))
        return B


# In[ ]:


# Derivatives of the activation functions

def derivative_activation(z,type):
    if(type=='sigmoid'):
        f=activation_function(z,'sigmoid')
        return f*(1-f)
    if(type=='tanh'):
        f=activation_function(z,'tanh')
        return 1-f*f
    if(type=='ReLu'):
        z[z <= 0] = 0
        z[z > 0] = 1
        return z
    if(type=='softmax'):
        return 1 


# In[ ]:


# Plots of the activation functions as well as their derivatives

t=np.arange(-6,6,0.01)

# Setup centered axes
fig, ax = plt.subplots(figsize=(9, 5))
ax.spines['left'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# Create and show plot
ax.plot(t,activation_function(t,'sigmoid'), color="#307EC7", linewidth=3, label="sigmoid")
ax.plot(t,derivative_activation(t,'sigmoid'), color="#9621E2", linewidth=3, label="derivative")
ax.legend(loc="upper right", frameon=False)
fig.show()

# ---------------------------------------------------------------------------------------

t=np.arange(-6,6,0.01)

# Setup centered axes
fig, ax = plt.subplots(figsize=(9, 5))
ax.spines['left'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# Create and show plot
ax.plot(t,activation_function(t,'tanh'), color="#307EC7", linewidth=3, label="tanh")
ax.plot(t,derivative_activation(t,'tanh'), color="#9621E2", linewidth=3, label="derivative")
ax.legend(loc="upper right", frameon=False)
fig.show()

# ---------------------------------------------------------------------------------------

t=np.arange(-6,6,0.01)

# Setup centered axes
fig, ax = plt.subplots(figsize=(9, 5))
ax.spines['left'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# Create and show plot
ax.plot(t,activation_function(t,'ReLu'), color="#307EC7", linewidth=3, label="ReLu")
ax.plot(t,derivative_activation(activation_function(t,'ReLu'),'ReLu'), color="#9621E2", linewidth=3, label="derivative")
ax.legend(loc="upper right", frameon=False)
fig.show()


# In[ ]:


# Converts the obtained probabilities into the output of the neural network

def max_one_hot(v):
    max_i=0
    val=v[0]
    for i in range(1,len(v)):
        if val<v[i]:
            max_i=i
            val=v[i]
    return max_i

max_one_hot(Out_Array(5))


# In[ ]:


# Initializing weights for the neural network

n_neurons=[785,201,11]
w=[]
for i in range(0,len(n_neurons)-1):
    a=np.random.randn(n_neurons[i+1],n_neurons[i])
    w.append(a)
w=np.true_divide(w,1000)


# In[ ]:


# Feed Forward of the inputs to obtain outputs
# Back Propagation of errors from the desired outputs for a single datapoint

def back_propagation(x_train, y_train, weights, n_neurons, act_list):
    
    val_neurons=[]
    net=[]
    errors=[]
    n=len(n_neurons)
    for i in range(0,n):
        a=np.zeros(n_neurons[i])
        net.append(a)
        val_neurons.append(a)
        errors.append(a)
    val_neurons[0]=x_train
    net[0]=x_train
    for i in range(1,n):
#         val_neurons[i]=(weights[i-1]@val_neurons[i-1])
        net[i]=weights[i-1]@val_neurons[i-1]
        val_neurons[i]=activation_function(net[i],act_list[i-1])
#         val_neurons[i]=net[i]
        val_neurons[i][n_neurons[i]-1]=1
        net[i][n_neurons[i]-1]=1

    val_neurons[n-1][n_neurons[n-1]-1]=0
    net[n-1][n_neurons[n-1]-1]=0
#     return val_neurons

#     Initializing errors
    for i in range(0,len(val_neurons)):
        for j in range(0,len(val_neurons[i])):
            errors[i][j]=0
    
    errors[n-1] = (Out_Array(y_train) - val_neurons[n-1])
    for i in range(n-2,0,-1):
        errors[i]=(derivative_activation(net[i],act_list[i-1])*(weights[i].T@errors[i+1]))
    errors[0]=np.array([])
    
    return errors, val_neurons, net


# b=back_propagation(x[0],y[0],w, [785,51,11], ['sigmoid', 'softmax'])
# b[0]
# w[0].shape
# np.array(b[0])
# back_propagation(np.ones(785),1,[np.ones((51,785)),np.ones((21,51)),np.ones((11,21))], [785,51,21,11], ['sigmoid', 'sigmoid', 'softmax'])[1]
# back_propagation(np.ones(785),1,w, [785,51,21,11], ['tanh', 'tanh', 'softmax'])
# f=np.array(f)
# f.shape

#         a=n_neurons[i-1]
#         b=n_neurons[i]
#         for j in range(0,b):
#             for k in range(0,n_neurons[i-1]):
#                 val_neurons[i][j]+=w[i-1][k][j]*val_neurons[i-1][k]


# In[ ]:


# Batch Gradient Descent of the weights for given neural network, iterations, learning rate, activations and regularization parameter

def gradient_descent(x_train, y_train, weights, n_neurons, act_list, ita, batch_size, itr, lam):
    n=len(x_train)
    epoch=itr*n//batch_size
    for i in range(0,epoch):
        batch_errors=[]
        batch_values=[]
        batch_net=[]
        for j in range(0,batch_size):
            b=back_propagation(x_train[(i*batch_size+j)%n], y_train[(i*batch_size+j)%n], weights, n_neurons, act_list)
            if j==0:
                batch_errors=b[0].copy()
                batch_values=b[1].copy()
                batch_net=b[2].copy()
            else:
                for m in range(0,len(n_neurons)):
                    batch_errors[m]=batch_errors[m]+b[0][m]
                    batch_values[m]=batch_values[m]+b[1][m]
                    batch_net[m]=batch_net[m]+b[2][m]
            for m in range(0,len(n_neurons)):
                batch_errors[m]=batch_errors[m]/batch_size
                batch_values[m]=batch_values[m]/batch_size
                batch_net[m]=batch_net[m]/batch_size
        for l in range(0,len(weights)-1):
            for k in range(0,n_neurons[l]):
                weights[l][:,k]=(1-lam)*weights[l][:,k]+ita*batch_values[l][k]*batch_errors[l+1]
                weights[l][:n_neurons[l+1]-1,k]=weights[l][:n_neurons[l+1]-1,k]+ita*batch_errors[l+1][n_neurons[l+1]-1]
#             weights[k]=weights[k]+ita*(batch_errors@batch_values)
    return weights
        
        

        
# gradient_descent(x, y, w, [785,51,11], ['sigmoid', 'softmax'], 0.001, 10, 4)


# In[ ]:


# Dividing Training Data into Train + Test segments

# upd_w = gradient_descent(x[:5000,:], y[:5000], w, [785,201,11], ['tanh', 'softmax'], 0.01, 10, 4)
# upd_w
# accuracy=0
# for i in range(5000,len(x)):
#     v=back_propagation(x[i],y[i],upd_w, [785,201,11], ['tanh', 'softmax'])[1]
#     if max_one_hot(v[len(upd_w)])==y[i]:
#         accuracy=accuracy+1
# accuracy/2000        
# # accuracy


# In[ ]:


# Finding Accuracy of the model for given activations

n_neurons=[785,201,11]
w=[]
for i in range(0,len(n_neurons)-1):
    a=np.random.randn(n_neurons[i+1],n_neurons[i])
    w.append(a)
w=np.true_divide(w,1000)


upd_w = gradient_descent(x, y, w, [785,201,11], ['ReLu', 'softmax'], 0.001, 2, 1, 0.000)
# upd_w = gradient_descent(x, y, upd_w1, [785,201,11], ['tanh', 'tanh', 'softmax'], 0.01, 10, 4)
accuracy=0
for i in range(0,len(x)):
    v=back_propagation(x[i],y[i],upd_w, [785,201,11], ['ReLu', 'softmax'])[1]
    if max_one_hot(v[len(upd_w)])==y[i]:
        accuracy=accuracy+1
accuracy/7000        


# In[ ]:


# Loss funtion

def Loss(x_train, y_train):
    upd_w = gradient_descent(x, y, w, [785,201,11], ['tanh', 'softmax'], 0.001, 4, 4, 0.000)
    n_neurons=[785,201,11]
    w1=[]
    for i in range(0,len(n_neurons)-1):
        a=np.random.randn(n_neurons[i+1],n_neurons[i])
        w1.append(a)
    w1=np.true_divide(w1,1000)
    accuracy=0
    cost=0
    initial_cost=0
    for i in range(0,len(x_train)):
        v=back_propagation(x[i],y[i],upd_w, [785,201,11], ['tanh', 'softmax'])[1]
        v1=back_propagation(x[i],y[i],w1, [785,201,11], ['tanh', 'softmax'])[1]
        if max_one_hot(v[len(upd_w)])==y[i]:
            accuracy=accuracy+1
        cost=cost+(np.log(v[len(upd_w)])*y[i])+((1-y[i])*np.log(1-v[len(upd_w)]))
#         initial_cost=initial_cost+(np.log(v1[len(upd_w)])*y[i])+((1-y[i])*np.log(1-v1[len(upd_w)]))-0.1
    return accuracy/len(x_train), np.sum(cost[:-1])/(len(x_train)*10), np.sum(initial_cost[:-1])/(len(x_train)*10)

Loss(x,y)


# In[ ]:


# upd_w1 = gradient_descent(x, y, w, [785,201,11], ['tanh', 'softmax'], 0.001, 4, 2)
# upd_w = gradient_descent(x, y, upd_w1, [785,201,11], ['tanh', 'softmax'], 0.001, 4, 2)


# In[ ]:


# Exporting the outputs of the neural network on the test data to a csv file

dftestoutput=pd.DataFrame(np.array([]))
# newee=pd.DataFrame(np.array([1]))
# dftestoutput=dftestoutput.append(newee)
for i in range(0,3000):
    v=back_propagation(x_test[i],y[i],upd_w, [785,201,11], ['tanh', 'softmax'])[1]
    newee=pd.DataFrame(np.array([max_one_hot(v[2])]))
    dftestoutput=dftestoutput.append(newee)
print(dftestoutput)
dftestoutput.to_csv('out.csv')


# In[ ]:


Batch_Size = [1,2,3,4,5,6,7,8,9,10]
l = []
for j in range(len(Batch_Size)):
    w=[]
    for i in range(0,len(n_neurons)-1):
        a=np.random.randn(n_neurons[i+1],n_neurons[i])
        w.append(a)
    w=np.true_divide(w,1000)
    upd_w = gradient_descent(x, y, w, [785,201,11], ['tanh', 'softmax'], 0.001, Batch_Size[j], 1, 0.000)
    # upd_w = gradient_descent(x, y, upd_w1, [785,201,11], ['tanh', 'tanh', 'softmax'], 0.01, 10, 4)
    accuracy=0
    for i in range(0,len(x)):
        v=back_propagation(x[i],y[i],upd_w, [785,201,11], ['tanh', 'softmax'])[1]
        if max_one_hot(v[len(upd_w)])==y[i]:
            accuracy=accuracy+1
    accuracy=accuracy/7000        
    l.append(accuracy)
    print(l[len(l)-1])
print(l)


# In[ ]:


epochs = [1,2,4,8,10]
l = []
for j in range(len(epochs)):
    w=[]
    for i in range(0,len(n_neurons)-1):
        a=np.random.randn(n_neurons[i+1],n_neurons[i])
        w.append(a)
    w=np.true_divide(w,1000)
    upd_w = gradient_descent(x, y, w, [785,201,11], ['tanh', 'softmax'], 0.005, 5, epochs[j], 0.000)
    # upd_w = gradient_descent(x, y, upd_w1, [785,201,11], ['tanh', 'tanh', 'softmax'], 0.01, 10, 4)
    accuracy=0
    for i in range(0,len(x)):
        v=back_propagation(x[i],y[i],upd_w, [785,201,11], ['tanh', 'softmax'])[1]
        if max_one_hot(v[len(upd_w)])==y[i]:
            accuracy=accuracy+1
    accuracy=accuracy/7000        
    l.append(accuracy)
    print(l[len(l)-1])
print(l)


# In[ ]:





# In[ ]:


learning_rate = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
accuracy = [0.4714 , 0.7637, 0.7998, 0.8627, 0.8717, 0.8137, 0.8053, 0.3658]

# Batch_Size = 2, Epochs = 1, Regularization = 0, [784,200,10], (tanh,softmax) activation

fig=plt.plot(learning_rate,accuracy)
plt.ylabel('Accuracy')
plt.xlabel('Learing rate')
plt.savefig('Accuracy vs Learning Rate.png')


# In[ ]:


Batch_Size1 = [1,5,10,50,100,500,1000,3500,7000]
accuracy1 = [0.8784285714285714, 0.7414285714285714, 0.32442857142857146, 0.08842857142857143, 0.12, 0.12414285714285714, 0.11428571428571428, 0.09785714285714285, 0.10942857142857143]

# Learning Rate = 0.001, Epochs = 1, Regularization = 0, [784,200,10], (tanh,softmax) activation

plt.plot(Batch_Size1,accuracy1)
plt.ylabel('Accuracy')
plt.xlabel('Batch Size')
plt.savefig('Accuracy vs Batch_Size1.png')


# In[ ]:


Batch_Size2 = [1,2,3,4,5,6,7,8,9,10]
accuracy2 = [0.8785714285714286, 0.8658571428571429, 0.8124285714285714, 0.7505714285714286, 0.7752857142857142, 0.731, 0.6505714285714286, 0.5558571428571428, 0.5054285714285714, 0.48328571428571426]
# Learning Rate = 0.001, Epochs = 1, Regularization = 0, [784,200,10], (tanh,softmax) activation

plt.plot(Batch_Size2,accuracy2)
plt.ylabel('Accuracy')
plt.xlabel('Batch Size')
plt.savefig('Accuracy vs Batch_Size2.png')


# In[ ]:


epochs = [1,2,4,8,10]
accuracy = [0.7938571428571428, 0.8314285714285714, 0.8621428571428571, 0.8638571428571429, 0.8655714285714285]
# Learning Rate = 0.005, Batch Size = 5, Regularization = 0, [784,200,10], (tanh,softmax) activation

plt.plot(epochs,accuracy)
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.savefig('Accuracy vs Epochs.png')


# In[ ]:


activations = ['sigmoid, softmax', 'ReLu, softmax', 'tanh, softmax']
accuracy = [84.93, 89.8, 83.56]
# Learning Rate = 0.005, Epoch = 1, Batch Size = 5, Regularization = 0, [784,200,10]

plt.plot(activations,accuracy)
plt.ylabel('Accuracy')
plt.xlabel('Activations')
plt.savefig('Accuracy vs Activations.png')


# In[ ]:


Number_of_neurons = [50, 100, 150, 170, 180, 200, 250, 300, 400, 500]
accuracy = [50.98, 9.31, 91.20, 91.27, 88.21, 88.54, 90.24, 90.7, 89.57, 91.04]
# Learning Rate = 0.09, Epoch = 1, Batch Size = 2, Regularization = 0, [784,200,10], (ReLu,softmax) activation

plt.plot(Number_of_neurons,accuracy)
plt.ylabel('Accuracy')
plt.xlabel('Number of Neurons in the Hidden Layer')
plt.savefig('Accuracy vs Number of Neurons.png')


# In[ ]:


a = ['Random Initialization', 'Zero Initialization']
b= [91.27,9.31]
plt.plot(a,b)
plt.ylabel('Accuracy')
plt.xlabel('Weights Initialization')
plt.savefig('Accuracy vs Weights Initialization.png')


# In[ ]:


# Hidden Layer Visualization
plt.imshow(upd_w[0][3][:-1].reshape(28,28))
plt.savefig('w03.png')
plt.imshow(upd_w[0][2][:-1].reshape(28,28))
plt.savefig('Visualization of weights of hidden layer.png')

