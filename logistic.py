
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import numpy as np

from sklearn.metrics import f1_score
import math

global X_train, X_test, y_train, y_test

# Load the data set as a float32 because my laptop can't even run the default64
dataset = pd.read_csv("WRITE DATA PATH").astype('float32')

# rename the first column to label which is the one that tells what letter this is 0-25 = A-Z
dataset.rename(columns={'0': 'label'}, inplace=True)

# Change the y or label column to letters for better understanding
alphabet_labels = []
for label in dataset['label']:
    alphabet_labels.append(chr(int(label) + 65))  # Convert numeric label to ASCII character to convert from 0-25 to A-Z

# Replace the 'label' column with the new alphabetic labels
#dataset['label'] = alphabet_labels

shuffled_dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True) # frac 1 to shuffle 100% and random state = 42 for reproducibility and reset index to reset index after shuffling

# Split the dataset into features (X) and target labels (y)
X = shuffled_dataset.drop('label', axis=1)  # Features are all columns except 'label'
y = shuffled_dataset['label']  # Target label which will be the letter

# show number of unqiue classes and their distribution
print("Number of unique classes:", y.nunique())

class_distribution = y.value_counts()
class_distribution.plot(kind='bar', figsize=(10, 6))
plt.title("Distribution of Classes")
plt.xlabel("Class (Letter)")
plt.ylabel("Number of Samples")
plt.show()

# Normalizing each image using minmax scaler
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data using MinMaxScaler after the split
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape flattened vectors into 28x28 2d images
X_reshaped = X_train.reshape(X_train.shape[0], 28, 28)
X_test_reshaped = X_test.reshape(X_test.shape[0], 28, 28)
print(X_reshaped.shape)

X_train_splited,X_Valid,y_train_splited,Y_Valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Add bias column of ones
X_train_splited = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_Valid = np.hstack([np.ones((X_Valid.shape[0], 1)), X_Valid])
X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

def compute_partial(x, y, h):
  m = len(y)
  res = (1 / m) * np.dot(x.T, (y - h))
  return res

def logistic(alpha, max_iter, x, y):
  theta = np.zeros(x.shape[1])
  for i in range(max_iter):
    # hypothesis function (sigmoid function of theta * x)
    h = 1 / (1 + np.exp(-np.dot(x, theta)))

    gradient = compute_partial(x, y, h)
    theta = theta - alpha * gradient
  return theta

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

m = len(y) # length of rows
def cost_function(theta, x, y):
  hypothesis = sigmoid(np.dot(x, theta)) # (x . theta) is the z of the sigmoid function
  cost = -(1 / m) * np.sum(y * np.log(hypothesis) + (1 - y) * np.log(1 - hypothesis)) # cost function as seen in lab 4, page 2
  return cost

def gradient_descent(alpha, max_iter, x, y):
  cost_history = []
  theta = np.zeros(x.shape[1])
  for i in range(max_iter):
    hypothesis = sigmoid(np.dot(x, theta))
    for j in range(theta.shape[0]):
      gradient = compute_partial(x, y, hypothesis)
      theta[j] = theta[j] - alpha * gradient[j]
    cost = cost_function(theta, x, y)
    cost_history.append(cost)
  return theta, cost_history

def logistic_train(alpha, max_iter, x, y, num_of_classes):
  rows, columns = x.shape
  all_weights = np.zeros((num_of_classes, columns))
  costs = []

  for i in range(num_of_classes):
    print(f"Training model {chr(i + 65)}")

    # gives rows equal to the current targeted class value of 1,
    # any other value 0
    y_binary = (y_train == i).astype(int)
    theta, cost_history = gradient_descent(alpha, max_iter, x, y)
    all_weights[i] = theta
    costs.append(cost_history)

  return all_weights, costs

def logistic_predict(x, all_weights):
  # multiplies the input values with the weights from training,
  # and normalizes it via sigmoid
  probabilities = sigmoid(np.dot(X, all_weights.T))

  predictions = np.argmax(probabilities, axis=1)  # Use argmax to get the highest possibility
  return predictions  # Return predicted classes

all_weights, costs = logistic_train(0.001, 2, X_train, y_train, 2)

def plot_cost_curves(costs):
  plt.figure(figsize=(10, 6))
  for i, cost_history in enumerate(costs):
      plt.plot(cost_history, label=f"Class {i}")
  plt.title("Error Curves for Each Class")
  plt.xlabel("Epochs")
  plt.ylabel("Cost")
  plt.legend()
  plt.show()

def validate_logistic(all_weights):
  y_val_prediction = logistic_predict(Y_Valid, all_weights)
  acc = np.mean(y_val_prediction == Y_Valid)
  return acc

y_test_prediction = logistic_predict(X_test, all_weights)

cm = confusion_matrix(y_test, y_test_prediction)
print("Confusion Matrix:")
print(cm)