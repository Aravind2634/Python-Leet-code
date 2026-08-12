# Simple ML Model Training - Single Page

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load dataset
data = load_iris()

X = data.data          # Features
y = data.target        # Labels

# 2. Split data into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Scale the data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Create ML model
model = LogisticRegression()

# 5. Train the model
model.fit(X_train, y_train)

# 6. Make predictions
predictions = model.predict(X_test)

# 7. Check accuracy
accuracy = accuracy_score(y_test, predictions)

print("Model Training Completed!")
print("Accuracy:", accuracy)

# 8. Test with a new flower
new_flower = [[5.1, 3.5, 1.4, 0.2]]

new_flower = scaler.transform(new_flower)

prediction = model.predict(new_flower)

print("Predicted Flower:",
      data.target_names[prediction[0]])
