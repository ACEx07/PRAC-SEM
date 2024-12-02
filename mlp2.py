import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

data = {
    'age': [25, 30, 35, 20, 40, 45, 22, 38, 50, 60],
    'income': [50000, 70000, 60000, 30000, 80000, 90000, 40000, 120000, 100000, 150000],
    'purchased': [0, 1, 1, 0, 1, 1, 0, 1, 1, 1]
}
df = pd.DataFrame(data)
X = df[['age', 'income']]
y = df['purchased']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))