import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score


# Load the data set as a float32 because my laptop can't even run the default64
dataset = pd.read_csv("A_Z Handwritten Data.csv").astype('float32')

# Rename the first column to label, which is the one that tells what letter this is (0-25 = A-Z)
dataset.rename(columns={'0': 'label'}, inplace=True)

# Change the y or label column to letters for better understanding
alphabet_labels = []
for label in dataset['label']:
    alphabet_labels.append(chr(int(label) + 65))  # Convert numeric label to ASCII character to convert from 0-25 to A-Z

# Replace the 'label' column with the new alphabetic labels
#dataset['label'] = alphabet_labels
shuffled_dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

# Split the dataset into features (X) and target labels (y)
X = shuffled_dataset.drop('label', axis=1)  # Features are all columns except 'label'
y = shuffled_dataset['label']  # Target label which will be the letter

# Show number of unique classes and their distribution
print("Number of unique classes:", y.nunique())

class_distribution = y.value_counts()
class_distribution.plot(kind='bar', figsize=(10, 6))
plt.title("Distribution of Classes")
plt.xlabel("Class (Letter)")
plt.ylabel("Number of Samples")
plt.show()

# Split the dataset into training, validation, and testing sets
# First split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Then split the training set into 80% training and 20% validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Normalize the data using MinMaxScaler after splitting
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Add a bias column of ones to all datasets
X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_val = np.hstack([np.ones((X_val.shape[0], 1)), X_val])
X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the cost function for logistic regression
def cost_function(theta, x, y):
    hypothesis = sigmoid(np.dot(x, theta))  # (x . theta) is the z of the sigmoid function
    cost = -(1 / len(y)) * np.sum(y * np.log(hypothesis) + (1 - y) * np.log(1 - hypothesis))
    return cost

# Gradient descent for logistic regression
def gradient_descent(alpha, max_iter, x, y):
    cost_history = []
    theta = np.zeros(x.shape[1])  # Initialize weights to zero
    for i in range(max_iter):
        hypothesis = sigmoid(np.dot(x, theta))
        gradient = (1 / len(y)) * np.dot(x.T, (hypothesis - y))
        theta -= alpha * gradient
        cost = cost_function(theta, x, y)
        cost_history.append(cost)
    return theta, cost_history

def logistic_train(alpha, max_iter, x, y, num_of_classes):
    rows, columns = x.shape
    all_weights = np.zeros((num_of_classes, columns))  # Store weights for each class
    costs = []

    for i in range(num_of_classes):
        print(f"Training model {chr(i + 65)}")

        # gives rows equal to the current targeted class value of 1,
        # any other value 0
        y_binary = (y == i).astype(int)
        theta, cost_history = gradient_descent(alpha, max_iter, x, y_binary)
        all_weights[i] = theta
        costs.append(cost_history)

    return all_weights, costs

def logistic_predict(x, all_weights):
    # multiplies the input values with the weights from training,
    # and normalizes it via sigmoid
    probabilities = sigmoid(np.dot(x, all_weights.T))
    predictions = np.argmax(probabilities, axis=1)  # Get the class with the highest probability
    return predictions

# Train the logistic regression model
num_of_classes = y.nunique()
all_weights, costs = logistic_train(0.4, 500, X_train, y_train, num_of_classes)
y_train_prediction = logistic_predict(X_train, all_weights)
train_accuracy = np.mean(y_train_prediction == y_train)
print(f"Training Accuracy: {train_accuracy:.4f}")

# Validate the model on the validation set
y_val_prediction = logistic_predict(X_val, all_weights)
val_accuracy = np.mean(y_val_prediction == y_val)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Evaluate the model on the test set
y_test_prediction = logistic_predict(X_test, all_weights)
cm = confusion_matrix(y_test, y_test_prediction)
print("Confusion Matrix:")
print(cm)

f1 = f1_score(y_test, y_test_prediction, average='weighted')
print(f"Average F1 Score: {f1:.4f}")

# Plot cost curves for all classes
def plot_cost_curves(costs):
    plt.figure(figsize=(10, 6))
    for i, cost_history in enumerate(costs):
        plt.plot(cost_history, label=f"Class {chr(i + 65)}")
    plt.title("Cost Curves for Each Class")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.legend()
    plt.show()

plot_cost_curves(costs)

# Plot accuracy curves for training and validation sets
train_accuracies = []
val_accuracies = []

for epoch in range(len(costs[0])):
    y_train_pred_epoch = logistic_predict(X_train, all_weights)
    y_val_pred_epoch = logistic_predict(X_val, all_weights)
    train_accuracies.append(np.mean(y_train_pred_epoch == y_train))
    val_accuracies.append(np.mean(y_val_pred_epoch == y_val))

plt.figure(figsize=(10, 6))
plt.plot(train_accuracies, label="Training Accuracy")
plt.plot(val_accuracies, label="Validation Accuracy")
plt.title("Accuracy Curves")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
