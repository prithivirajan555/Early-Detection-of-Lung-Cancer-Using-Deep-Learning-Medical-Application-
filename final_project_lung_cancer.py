import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

data1 = pd.read_csv('Lung_Cancer_Dataset.csv')
data = pd.read_csv('Lung_Cancer_Dataset.csv')

print(data.head())
shape=data.shape
print(shape)
print("---------")


col_names=data.columns
print(col_names)
print("---------")

numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns

for column in numeric_columns:
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    data = data[(data[column] >= Q1 - 1.5 * IQR) & (data[column] <= Q3 + 1.5 * IQR)]

print(data.head())
data = data.drop_duplicates()
LC = data['LUNG_CANCER'].value_counts()
labels = LC.index.tolist()
sizes = LC.tolist()
plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['pink', 'yellow'])
plt.title('Lung Cancer Distribution')
plt.axis('equal')
plt.show()
print("-----")

data_train = data.copy()

data_train['GENDER'] = data_train.GENDER.map({'M':1,'F':2})
data_train['LUNG_CANCER'] = data_train.LUNG_CANCER.map({'YES':2,'NO':1})
data_train
x = data_train.drop("LUNG_CANCER", axis=1)
y = data_train["LUNG_CANCER"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.17, random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
import joblib


imputer = SimpleImputer(strategy='mean')
clf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=15, min_samples_leaf=10, random_state=42)
pipeline = Pipeline(steps=[('imputer', imputer), ('classifier', clf)])

cv_scores = cross_val_score(pipeline, x_train, y_train, cv=10, scoring='accuracy')

print(f'Cross-Validation Accuracy (10-fold): {np.mean(cv_scores) * 100:.2f}%')

pipeline.fit(x_train, y_train)

y_pred = pipeline.predict(x_test)

y_train_pred = pipeline.predict(x_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_pred)

accuracies = ['Training', 'Testing']
values = [train_accuracy, test_accuracy]

plt.figure(figsize=(8, 6))
plt.bar(accuracies, values, color=['blue', 'yellow'])
plt.title('Training vs Testing Accuracy')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.show()

conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(y_test.reset_index(drop=True), label='Actual', marker='o')
plt.plot(pd.Series(y_pred).reset_index(drop=True), label='Predicted', marker='x')
plt.title('Actual vs Predicted Values')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.legend()
plt.show()