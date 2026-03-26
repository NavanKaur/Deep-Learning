#working with MNIST dataset
#import the libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report , confusion_matrix
from tensorflow.keras.datasets import mnist

np.random.seed(42)

#load and prepare the dataset

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0

X_train=X_train.reshape(-1 , 784)
X_test=X_test.reshape(-1, 784)

enc=OneHotEncoder(sparse_output=False)
y_train=enc.fit_transform(y_train.reshape(-1,1))
y_test=enc.transform(y_test.reshape(-1,1))

#visualization 1: Sample Images

plt.figure(figsize=(6,6))
for i in range (9):
    plt.subplot(3,3,i+1)
    plt.imshow(X_train[i].reshape(28,28),cmap="gray")
    plt.title(f"Label:{np.argmax(y_train[i])}")
    plt.axis("off")
plt.suptitle("Samle MNIST Images")
plt.show()

#train validation split

val_size=10000
X_val=X_train[:val_size]
y_val=y_train[:val_size]

X_train_small=X_train[val_size:]
y_train_small=y_train[val_size:]

#experimantaion modes
MODE="good_fit"

configs={
    "underfitting": {"hidden_size":16 , "lr":0.1 , "epochs":20},
    "good_fit": {"hidden_size":64 , "lr":0.1 , "epochs":50},
    "overfitting": {"hidden_size":128 , "lr":0.1 , "epochs":150},
    "slow_learning": {"hidden_size":64 , "lr":0.1 , "epochs":50}
}
config=configs[MODE]

hidden_size=config["hidden_size"]
lr=config["lr"]
epochs=config["epochs"]

print(f"\nRunning Mode:{MODE}")
print(config)

#model init

input_size=784
output_size=10

#He Initilization
w1=np.random.randn(input_size,hidden_size) * np.sqrt(2/input_size)
w2=np.random.randn(hidden_size,output_size) * np.sqrt(2/hidden_size)

#bias 
b1=np.zeros((1,hidden_size))
b2=np.zeros((1,output_size))

#Activations
def relu(x):
    return np.maximum(0,x)

def softmax(x):
    exp=np.exp(x - np.max(x,axis=1,keepdims=True))
    return exp / exp.sum(axis=1 , keepdims=True)

#forward
def forward(X):
    z1 = np.dot(X, w1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

#loss
def compute_loss(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))

##backward(fixed)
def backward(X, y, z1, a1, z2, a2):
    m = X.shape[0]

    dz2 = a2 - y
    dw2 = np.dot(a1.T, dz2) / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m

    dz1 = np.dot(dz2, w2.T) * (z1 > 0)  # ReLU derivative
    dw1 = np.dot(X.T, dz1) / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m

    return dw1, dw2, db1, db2

#training
losses = []
val_losses = []

for epoch in range(epochs):

    z1, a1, z2, a2 = forward(X_train_small)

    loss = compute_loss(y_train_small, a2)
    losses.append(loss)

    _, _, _, val_pred = forward(X_val)
    val_loss = compute_loss(y_val, val_pred)
    val_losses.append(val_loss)

    dw1, dw2, db1, db2 = backward(X_train_small, y_train_small, z1, a1, z2, a2)

    w1 -= lr * dw1
    w2 -= lr * dw2
    b1 -= lr * db1
    b2 -= lr * db2

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Train Loss: {loss:.4f} | Val Loss: {val_loss:.4f}")
        
#loss visualization
plt.figure()
plt.plot(losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.legend()
plt.title(f"Optimization vs Generalization ({MODE})")
plt.show()

#generalization gap
gap = np.array(val_losses) - np.array(losses)

plt.figure()
plt.plot(gap)
plt.title("Generalization Gap")
plt.xlabel("Epoch")
plt.ylabel("Gap")
plt.show()

#evaluation

_, _, _, preds = forward(X_test)

y_pred = np.argmax(preds, axis=1)
y_true = np.argmax(y_test, axis=1)

accuracy = np.mean(y_pred == y_true)
print("\nTest Accuracy:", accuracy)

#metrics
print("\nClassification Report:")
print(classification_report(y_true, y_pred))

cm = confusion_matrix(y_true, y_pred)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.show()

##predictions
plt.figure(figsize=(6,6))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[i].reshape(28,28), cmap='gray')
    plt.title(f"P:{y_pred[i]} | T:{y_true[i]}")
    plt.axis('off')
plt.show()

#interpretation
print("\nINTERPRETATION:")

if MODE == "underfitting":
    print("→ Model too simple → cannot learn patterns")
elif MODE == "overfitting":
    print("→ memorizing training data")
elif MODE == "good_fit":
    print("→ balanced learning")
elif MODE == "slow_learning":
    print("→ learning too slow")

print("\nKEY IDEA:")
print("Optimization = minimizing loss")
print("Generalization = performing on unseen data")