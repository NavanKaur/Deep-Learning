#task: Solving the XOR problem 

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

X=np.array([[0,0],
            [0,1],
            [1,0],
            [1,1]])

y=np.array([[0],
            [1],
            [1],
            [0]])

print("Input :\n",X)
print("Output: \n",y)

# Visulaizing the problem
plt.scatter(X[:,0],X[:,1], c=y.flatten(),cmap="coolwarm")
plt.title("XOR Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

#Initialize MLP Parametre

"""
input neurons  = 2
hidden neurons = 3
output neuron  = 1
"""
np.random.seed(42)

input_size=2
hidden_size=3
output_layer=1

W1=np.random.randn(input_size,hidden_size)
b1=np.zeros((1,hidden_size))

W2= np.random.randn(hidden_size,output_layer)
b2=np.zeros((1,output_layer))

print("W1: \n",W1)
print("W2: \n",W2)

#Visualizing: Weights as heatmap
plt.imshow(W1 , aspect="auto",cmap="Set3")
plt.colorbar()
plt.title("Weights : Input -> Hidden (W1)")
plt.xlabel("Hidden Neurons")
plt.ylabel("Input Features")
plt.show()

plt.imshow(W2 , aspect="auto")
plt.colorbar()
plt.title("Weights : Hidden -> Output (W2)")
plt.xlabel("Output Neurons")
plt.ylabel("Hidden Neurons")
plt.show()

#Defining the Activation Function(Sigmoid Function)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#forward propagation 
def forward(X):
    z1=np.dot(X,W1) + b1
    a1=sigmoid(z1)
    
    z2=np.dot(a1,W2) +b2
    a2= sigmoid(z2)
    
    return z1 , a1 ,z2 , a2

# run forward pass

z1 , a1 , z2 , output = forward(X)

print("Hidden Layer Output: \n",a1)
print("Final Output :\n",output)

# visualizing : Hidden Layer Activations

plt.imshow(a1 , aspect="auto")
plt.colorbar()
plt.title("Hidden Layers Activations")
plt.xlabel("Hidden Neurons")
plt.ylabel("Samples")
plt.show()

#Understanding Data flow

for i in range (len(X)):
    print(f"\nInput : {X[i]}")
    print(f"Hidden Activations : {a1 [i]}")
    print(f"Output : {output[i]}")
    
# Visualizing : Predictions vs Actual (Before Training)
plt.bar(range(len(y)) , y.flatten() , alpha=0.6, label="Actual")
plt.bar(range(len(output)) , output.flatten() , alpha=0.6, label="Predicted")

plt.legend()
plt.title("Before Training: Predictions vs Actual")
plt.show()

loss = np.mean((y - output) ** 2)
print("Loss : ",loss)

#Training with Loss tracking
learning_rate=0.1
losses=[]

for epoch in range (10000):
    z1,a1,z2,output = forward(X)
    
    loss=np.mean(( y - output ) ** 2)
    losses.append(loss)
    
    error= y - output
    
    d_output= error * output * (1-output)
    d_hidden= d_output.dot(W2.T) * a1 * (1-a1)
    
    W2 += a1.T.dot(d_output) * learning_rate
    W1 += X.T.dot(d_hidden) * learning_rate
    
#visualizing : Loss Curve

plt.plot(losses)
plt.title("Loss Decreasing Over Time") 
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

_,_,_, predictions= forward(X)
print("Predcitons: \n",predictions)

#Visualizing : Predcitons After Training
plt.bar(range(len(y)) , y.flatten() , alpha=0.6, label="Actual")
plt.bar(range(len(predictions)) , predictions.flatten() , alpha=0.6, label="Predicted")
plt.legend()
plt.title("After Training: Predictions vs Actual")
plt.show()

#Decision Boundary 
xx, yy = np.meshgrid(np.linspace(-0.5,1.5,100),
                     np.linspace(-0.5,1.5,100))

grid = np.c_[xx.ravel() , yy.ravel()]
_,_,_, preds= forward(grid)

Z= preds.reshape(xx.shape)

plt.contourf(xx,yy,Z,aplha=0.5)
plt.scatter(X[:,0],X[:,1],c=y.flatten(),edgecolors='k')
plt.title("Decision Boundary")
plt.show()