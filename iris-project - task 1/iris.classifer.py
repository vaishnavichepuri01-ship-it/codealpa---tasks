import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#1.LOAD THE DATASET
df =pd.read_csv("iris.csv")
print(df.head())
print(df.shape)
print(df.columns)
print(df.info())
print(df["Species"].value_counts())
#2.SELECT FEATIRES AND TARGET
X = df[["SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm", 
        "PetalWidthCm"]]
Y = df["Species"]
#3. SPLIT THE DATA
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)
#4.LOGISTIC REGRESSION MODEL
model = LogisticRegression()
model.fit(X_train, Y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(Y_test, predictions)
print("ACCURACY VALUE =", accuracy)
cm = confusion_matrix(Y_test, predictions)
print("Confusion Matrix:")
print(cm)
print("Classification Report:")
print(classification_report(Y_test, predictions))
#5.KNN MODEL
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)
knn_predictions = knn.predict(X_test)
knn_accuracy = accuracy_score(Y_test, knn_predictions)
print("KNN Accuracy:", knn_accuracy)
knn_cm = confusion_matrix(Y_test, knn_predictions)
print("KNN Confusion Matrix:")
print(knn_cm)
#6. MODEL COMPARISON
print("\nModel Comparison")
print("Logistic Regression:", accuracy)
print("KNN:", knn_accuracy)
#7. PREDICTION FOR NEW DATA
sepal_length = float(input("Enter Sepal Length: "))
sepal_width = float(input("Enter Sepal Width: "))
petal_length = float(input("Enter Petal Length: "))
petal_width = float(input("Enter Petal Width: "))

new_flower = [[
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
]]

prediction = model.predict(new_flower)

print("Predicted Species:", prediction[0])

