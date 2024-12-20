from main import neural_networks, y_test,X_test_reshaped
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from sklearn.metrics import f1_score


model1 = neural_networks([256, 128, 64, 26], 'sigmoid')
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