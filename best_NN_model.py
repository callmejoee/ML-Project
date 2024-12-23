from main import neural_networks, y_test,X_test_reshaped, scaler, visualize_predictions
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from PIL import Image


model1 = neural_networks([256, 128, 64, 26], 'softmax')
y_predict = model1.predict(X_test_reshaped)

y_pred_classes = np.argmax(y_predict, axis=1)
cm = confusion_matrix(y_test, y_pred_classes)
print("Confusion_Matrix:")
print(cm)

f1_macro = f1_score(y_test, y_pred_classes, average='macro')
f1_micro = f1_score(y_test, y_pred_classes, average='micro')
f1_weighted = f1_score(y_test, y_pred_classes, average='weighted')

print("Macro-average F1-Score:", f1_macro)
print("Micro-average F1-Score:", f1_micro)
print("Weighted-average F1-Score:", f1_weighted)

print("First Model:")
neural_networks([128, 64, 26], 'softmax')
print("Second Model")
neural_networks([256, 128, 64, 26], 'relu')

def preprocess_letter_image(image_path):
    image = Image.open(image_path)

    image = image.convert('L')
    image = image.resize((28, 28))
    #image.show()

    img_array = np.array(image)
    img_array = img_array.reshape(1, -1)
    img_array = scaler.transform(img_array)
    img_array = img_array.reshape(1, 28, 28)

    img_array = np.clip(img_array, 0, 1)

    img_array = (img_array * 255).astype(np.uint8)
    #img_array = 255 - img_array

    return img_array

def visualize_test_results(image_paths, model):
    plt.figure(figsize=(10, 10))
    for i, image_path in enumerate(image_paths):
        processed_image = preprocess_letter_image(image_path)
        letter_predict = model.predict(processed_image)
        letter_pred_classes = np.argmax(letter_predict, axis=1)
        letter = chr(letter_pred_classes[0]+65)

        plt.subplot(5, 5, i + 1)
        shown_image = processed_image.squeeze()
        plt.imshow(shown_image, cmap='Greys')
        plt.title(f"Pred: {letter}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Example usage(Team Member_1)
image_paths = ["/content/drive/MyDrive/ML Project/M.jpeg", "/content/drive/MyDrive/ML Project/A.jpeg","/content/drive/MyDrive/ML Project/R.jpeg", "/content/drive/MyDrive/ML Project/K.jpeg"]
visualize_test_results(image_paths, model1)

# Example usage(Team Member_2)
image_paths = ["/content/drive/MyDrive/ML Project/O.jpeg", "/content/drive/MyDrive/ML Project/M.jpeg","/content/drive/MyDrive/ML Project/A.jpeg", "/content/drive/MyDrive/ML Project/R.jpeg"]
visualize_test_results(image_paths, model1)

# Example usage(Team Member_3)
image_paths = ["/content/drive/MyDrive/ML Project/Y.jpeg", "/content/drive/MyDrive/ML Project/O.jpeg","/content/drive/MyDrive/ML Project/U.jpeg", "/content/drive/MyDrive/ML Project/S.jpeg","/content/drive/MyDrive/ML Project/S.jpeg","/content/drive/MyDrive/ML Project/E.jpeg","/content/drive/MyDrive/ML Project/F.jpeg"]
visualize_test_results(image_paths, model1)

# Example usage(Team Member_4)
image_paths = ["/content/drive/MyDrive/ML Project/H.jpeg", "/content/drive/MyDrive/ML Project/A.jpeg","/content/drive/MyDrive/ML Project/B.jpeg", "/content/drive/MyDrive/ML Project/I.jpeg","/content/drive/MyDrive/ML Project/B.jpeg","/content/drive/MyDrive/ML Project/A.jpeg"]
visualize_test_results(image_paths, model1)

#visualize_predictions(X_test_reshaped, y_test, y_pred_classes)