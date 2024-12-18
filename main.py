import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
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
dataset['label'] = alphabet_labels

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

scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)


# Reshape flattened vectors into 28x28 2d images
X_reshaped = X_normalized.reshape(X_normalized.shape[0], 28, 28) # shape 0 is for the number of rows ex. (100000, 784)

# Display a few images
plt.figure(figsize=(10, 10))
for i in range(16):  # Display random 16 images
    plt.subplot(4, 4, i + 1)
    plt.imshow(X_reshaped[i], cmap='Greys', interpolation='nearest') #image show function to display 2d arrays as images
    plt.axis('off')
plt.show()


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)
print("hello")
def svm_linear_model():
    print("hello")
    global X_train, X_test, y_train, y_test
    model= SVC(kernel="linear",C=2)
    model.fit(X_train,y_train)
    y_pre=model.predict(X_test)
    cm=confusion_matrix(y_test,y_pre)
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix for SVM (Linear Kernel)")
    plt.show()

# svm_linear_model()
def svm_nonlinear_model():
    global X_train, X_test, y_train, y_test
    model = SVC()
    model.fit(X_train, y_train)
    y_pre = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pre)
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix for SVM (Linear Kernel)")
    plt.show()
svm_nonlinear_model()