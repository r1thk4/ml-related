import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge

# Load diabetes dataset (using all features)
diabetes = load_diabetes()
X = diabetes.data  # All features
y = diabetes.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Custom Ridge Regression implementation (Vectorized)
class RidgeRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, lambda_=0.1):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.lambda_ = lambda_
    
    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape
        self.weights = np.zeros(self.n_features)
        self.bias = 0
        
        for _ in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (-2 * np.dot(X.T, (y - y_pred)) + 2 * self.lambda_ * self.weights) / self.n_samples
            db = -2 * np.sum(y - y_pred) / self.n_samples
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Train custom Ridge model
lambda_ = 12.5
custom_model = RidgeRegression(learning_rate=0.01, n_iterations=1000, lambda_=lambda_)
custom_model.fit(X_train, y_train)
custom_y_pred = custom_model.predict(X_test)

# Sklearn Linear Regression (no regularization)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_y_pred = lr_model.predict(X_test)

# Sklearn Ridge Regression
ridge_model = Ridge(alpha=lambda_)
ridge_model.fit(X_train, y_train)
ridge_y_pred = ridge_model.predict(X_test)

# Evaluate all models
def evaluate_model(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{name} Evaluation:")
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")

evaluate_model("Custom Ridge Regression", y_test, custom_y_pred)
evaluate_model("Sklearn Linear Regression", y_test, lr_y_pred)
evaluate_model("Sklearn Ridge Regression", y_test, ridge_y_pred)

# Compare weights
print("\nDifference in Weights (Custom Ridge vs Sklearn Ridge):")
print(np.abs(custom_model.weights - ridge_model.coef_))

print("\nDifference in Bias (Custom Ridge vs Sklearn Ridge):")
print(np.abs(custom_model.bias - ridge_model.intercept_))

# Plotting Predicted vs Actual for all models
plt.figure(figsize=(8, 6))
plt.scatter(custom_y_pred, y_test, label='Custom Ridge', alpha=0.7)
plt.scatter(ridge_y_pred, y_test, label='Sklearn Ridge', alpha=0.7)
plt.scatter(lr_y_pred, y_test, label='Sklearn Linear', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal Line')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Predicted vs Actual for Different Models')
plt.legend()
plt.grid(True)
plt.show()
