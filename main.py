import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

import tensorflow as tf
import numpy as np

from sklearn.metrics import f1_score

global X_train, X_test, y_train, y_test

# Load the data set as a float32 because my laptop can't even run the default64
dataset = pd.read_csv("A_Z Handwritten Data.csv").astype('float32')

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
#plt.show()
# Normalizing each image using minmax scaler
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data using MinMaxScaler after the split
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape flattened vectors into 28x28 2d images
X_reshaped = X_train.reshape(X_train.shape[0], 28, 28)
X_text_reshaped = X_test.reshape(X_test.shape[0], 28, 28)
print(X_reshaped.shape)

# Display a few images
plt.figure(figsize=(10, 10))
for i in range(20):  # Display random 16 images
    plt.subplot(5, 4, i + 1)
    plt.imshow(X_reshaped[i], cmap='Greys', interpolation='nearest') #image show function to display 2d arrays as images
    plt.axis('off')
plt.show()
def visualize_predictions(X_test, y_test, y_pred):
    num_samples = 20
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    X_samples = X_test[indices].reshape(num_samples, 28, 28)
    y_samples_true = y_test.iloc[indices].values
    y_samples_pred = y_pred[indices]
    plt.figure(figsize=(10, 10))
    for i in range(num_samples):
        plt.subplot(5, 4, i + 1)
        plt.imshow(X_samples[i], cmap='Greys', interpolation='nearest')
        plt.title(f"True: {y_samples_true[i]}\nPred: {y_samples_pred[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def svm_linear_model():
    global X_train, X_test, y_train, y_test
    model= SVC(kernel="linear",C=2)
    model.fit(X_train,y_train)
    y_pre=model.predict(X_test)
    cm=confusion_matrix(y_test,y_pre)
    print(cm)
    print("\nconfusion_matrix Linear SVM:")
    report = classification_report(y_test, y_pre)
    print("\nClassification Report Linear SVM:")
    print(report)
    #visualize_predictions(X_test, y_test, y_pre)

#svm_linear_model()

# def svm_nonlinear_model():
#     global X_train, X_test, y_train, y_test
#     model = SVC()
#     model.fit(X_train, y_train)
#     y_pre = model.predict(X_test)
#     cm = confusion_matrix(y_test, y_pre)
#     print(cm)
#     f1 = f1_score(y_test, y_pre, average='weighted')
#     print(f"Average F1 Score Non-Linear SVM: {f1:.4f}")
#     visualize_predictions(X_test, y_test, y_pre)
#
# svm_nonlinear_model()

def build_model(neurons_of_layers, activation):
  parameters_arr = [
      tf.keras.layers.Flatten(input_shape=(28, 28)),
  ]

  for num_of_neurons in neurons_of_layers[:-1]:
    parameters_arr.append(tf.keras.layers.Dense(num_of_neurons, activation=activation))
  parameters_arr.append(tf.keras.layers.Dense(neurons_of_layers[-1]))

  model = tf.keras.Sequential(parameters_arr)
  return model

def setup_model(model):
  model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

def neural_networks():
  model1 = build_model([128, 64, 26], 'relu')
  setup_model(model1)
  model1.fit(X_reshaped, y_train, epochs=3)
  test_loss, test_acc = model1.evaluate(X_text_reshaped,  y_test, verbose=2)
  print('\nTest accuracy1:', test_acc)
  model2 = build_model([256, 128, 64, 26], 'sigmoid')
  setup_model(model2)
  model2.fit(X_reshaped, y_train, epochs=3)
  test_loss, test_acc = model2.evaluate(X_text_reshaped, y_test, verbose=2)

  print('\nTest accuracy2:', test_acc)

neural_networks()
print(X_test.shape)
print(y_train.shape)