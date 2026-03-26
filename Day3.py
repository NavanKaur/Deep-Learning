#Practising the Backpropagation

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay


#load the dataset
try :
    data=pd.read_csv("heart2(DL).csv")
    print("Dataset loaded successfully")
except FileNotFoundError:
    print("Dataset Not loaded ")
    
print("First few rows :\n",data.head())
print("dataset information:")
print(data.info())
print("dataset statistical summary:\n",data.describe())

# #feature distribution
# #histogram
# data.hist(figsize=(12,8))
# plt.suptitle("feature Distribution (before normalization)")
# plt.show()

# #pairplot
# sns.pairplot(data=data)
# plt.show()

# #Correlation heatmap
# sns.heatmap(data.corr(),annot=True)
# plt.title("Correlation heatmap")
# plt.show()

#get the names of  the columns
print(data.columns)

#normalize the dataset
data_norm=data[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal']].apply(lambda x: (x - x.min()) / (x.max()- x.min()))

print("normalised dataset:",data_norm.describe())

#before vs after normalization comparison

fig , ax =plt.subplots(1,2 , figsize=(12,4))

ax[0].boxplot(data[['age','chol','thalach']].values)
ax[0].set_title("Before Normalization")

ax[1].boxplot(data_norm[['age','chol','thalach']].values)
ax[1].set_title("After Normalization")

plt.show()

#create the target column to have integer 0/1
target=data['target']
print(target.sample(n=5))

#add the target column to the normalized datset
data_norm=pd.concat([data_norm,target],axis=1)
print(data_norm.sample(n=5))

#mark some data to test as unseen data
train_test_per= 75/100.0
data_norm['train']=np.random.rand(len(data_norm)) < train_test_per
print(data_norm.sample(n=5))

#separate train data
train=data_norm[data_norm.train==1]
train=train.drop('train',axis=1).sample(frac=1)
print(train.sample(n=5))

#seaparte test data
test=data_norm[data_norm.train==0]
test=test.drop('train',axis=1)
print(train.sample(n=5))

X=train.values[:,:13]
print(X[:14])

targets=[[1,0],[0,1]]
y=np.array([targets[int(x)] for x in train.values[:,13:14]])
print(y[:14])

#create backpropgation neural network
num_inputs = len(X[0])
hidden_layer_neurons = 14
np.random.seed(13)
w1 = 2*np.random.random((num_inputs, hidden_layer_neurons)) - 1
print(w1)

#connect hidden layer and input layer
num_outputs = len(y[0])
w2 = 2*np.random.random((hidden_layer_neurons, num_outputs)) - 1
print(w2)

#Neural Network Architecture Visualization
fig = plt.figure()
ax = fig.gca()
ax.axis('off')
plt.title("Neural Network Architecture")
plt.show()

#train the neural network by updating weights
def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k')
                ax.add_artist(line)
                
learning_rate = 0.28 # slowly update the network
losses = []

for epoch in range(50000):
    l1 = 1/(1 + np.exp(-(np.dot(X, w1)))) # sigmoid function
    l2 = 1/(1 + np.exp(-(np.dot(l1, w2))))
    
    er = (abs(y - l2)).mean()
    losses.append(er)
    
    l2_delta = (y - l2)*(l2 * (1-l2))
    l1_delta = l2_delta.dot(w2.T) * (l1 * (1-l1))
    
    w2 += l1.T.dot(l2_delta) * learning_rate
    w1 += X.T.dot(l1_delta) * learning_rate

    # 🔹 Visualization checkpoints (every 10000 epochs)
    if epoch % 10000 == 0:
        print(f"Epoch: {epoch}, Error: {er}")
        
        # 1. Hidden layer activation distribution
        plt.figure()
        plt.hist(l1.flatten(), bins=30)
        plt.title(f"Hidden Layer Activations (Epoch {epoch})")
        plt.xlabel("Activation Value")
        plt.ylabel("Frequency")
        plt.show()
        
        # 2. Output prediction distribution
        plt.figure()
        plt.hist(l2.flatten(), bins=30)
        plt.title(f"Output Predictions Distribution (Epoch {epoch})")
        plt.xlabel("Prediction Value")
        plt.ylabel("Frequency")
        plt.show()
        
        # 3. Loss curve so far
        plt.figure()
        plt.plot(losses)
        plt.title(f"Loss Curve (up to Epoch {epoch})")
        plt.xlabel("Epoch")
        plt.ylabel("Error")
        plt.show()

print('Error:', er)


# 🔹 Final Loss Curve (clean view)
plt.figure()
plt.plot(losses)
plt.title("Final Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.show()

#test the network for accuracy
X = test.values[:,:13]
y = np.array([targets[int(x)] for x in test.values[:,13:14]])

l1 = 1/(1 + np.exp(-(np.dot(X, w1))))
l2 = 1/(1 + np.exp(-(np.dot(l1, w2))))

np.round(l2,3)

plt.figure()
plt.hist(l2.flatten(), bins=30)
plt.title("Prediction Probabilities Distribution")
plt.xlabel("Confidence")
plt.ylabel("Frequency")
plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(np.argmax(y, axis=1), np.argmax(l2, axis=1))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.show()