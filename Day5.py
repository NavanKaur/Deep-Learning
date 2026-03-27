#Using fashion MNIST dataset

#importing libraries
import numpy as np
import matplotlib.pyplot as  plt
from tensorflow import keras
from tensorflow.keras import layers

np.random.seed(42)

#load fashion MNIST datatset
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

#normalize (0 to 255 becomes 0 to 1)

x_train = x_train / 255.0
x_test = x_test / 255.0

#visualize sample images
plt.figure(figsize=(6,6))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(x_train[i], cmap="gray")
    plt.title(f"Label: {y_train[i]}")
    plt.axis("off")

plt.suptitle("Sample Clothing Images")
plt.show()

#check class balance 
unique , counts = np.unique(y_train , return_counts=True)

plt.bar(unique,counts, edgecolor="black")
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

#Step1:define neural network architecture

model = keras.Sequential([
    
    #flatten converts 2d image to 1d vector
    layers.Flatten(input_shape=(28,28)),
    
    #Hidden layer 
    layers.Dense(128, activation="relu"),
    
    #output layer - 10 classes
    layers.Dense(10,activation="softmax")
    
])

#step2: compile the model
model.compile(
    optimizer="adam" , #adaptive learning
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Step3: fit the model
history= model.fit(
    x_train , y_train ,
    epochs=15,
    validation_split=0.2,
    batch_size=32
)

plt.plot(history.history['loss'], label = "Train Loss")
plt.plot(history.history["val_accuracy"],label="Validation Accuracy")
plt.legend()
plt.title("Accuracy Curve")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()

test_loss , test_acc =model.evaluate(x_test , y_test)
print("Test Accuracy :",test_acc)

preds=model.predict(x_test[:9])

plt.figure(figsize=(6,6))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(x_test[i],cmap="gray")
    plt.title(f"P:{np.argmax(preds[i])} | T: {y_test[i]}")
    plt.axis("off")
plt.suptitle("Predictions vs Actual")
plt.show()

from sklearn.metrics import confusion_matrix
import seaborn as sns
y_pred=np.argmax(model.predict(x_test),axis=1)
cm=confusion_matrix(y_test , y_pred )

sns.heatmap(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()