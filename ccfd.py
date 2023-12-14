import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# load data
location = r"C:\Users\Dell\Downloads\archive (1)\creditcard.csv"
column_names = ['V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,V21,V22,V23,V24,V25,V26,V27,V28,amount,class']  # Set your column names if the file doesn't have a header
data = pd.read_csv(location, header=None, names=column_names)
# divide data into features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# apply KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# apply Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)

# calculate accuracy
accuracy_knn = accuracy_score(y_test, y_pred_knn)
accuracy_gnb = accuracy_score(y_test, y_pred_gnb)

print(f'KNN Accuracy: {accuracy_knn}')
print(f'GNB Accuracy: {accuracy_gnb}')

# calculate confusion matrix
confusion_matrix_knn = confusion_matrix(y_test, y_pred_knn)
confusion_matrix_gnb = confusion_matrix(y_test, y_pred_gnb)

print(f'KNN Confusion Matrix: \n{confusion_matrix_knn}')
print(f'GNB Confusion Matrix: \n{confusion_matrix_gnb}')