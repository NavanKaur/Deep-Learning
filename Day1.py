import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

np.random.seed(42)
n=200

#interger values
study_hours=np.random.randint(1,11,n)
sleep_hours=np.random.randint(4,11,n)

#rule for pass or fail
#if effort is good then pass
score=study_hours * 0.7 +sleep_hours* 0.3

#convert to binary
result=(score > 6).astype(int)

data = pd.DataFrame ({
    "Study Hours": study_hours,
    "Sleep Hours": sleep_hours,
    "result": result
})


print(data.head())

#visualize the data 
plt.scatter(data["Study Hours"] , data["Sleep Hours"] , c=data["result"],edgecolors="black")
plt.xlabel("Study Hours")
plt.ylabel("Sleep Hours")
plt.title("Pass or Fail")
plt.show()

#define the neuron 
X=data[["Study Hours", "Sleep Hours"]].values
y=data["result"].values

#initiial weights
w= np.array([0.5 ,0.3])
b=-2

#sigmoid function  
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def neuron (X, w, b):
    z=np.dot (X,w) + b
    return sigmoid(z)

y_pred_prob=neuron(X , w, b)

y_pred= (y_pred_prob > 0.5).astype(int)

accuracy = np.mean (y_pred == y)
print("Accuracy:",accuracy)

plt.scatter(range(len(y)) , y , label="Actual ",alpha=0.6)
plt.scatter(range(len(y)) , y_pred , label="Predicted",alpha=0.6)
plt.legend()
plt.title("Actual vs Predicted")
plt.show()