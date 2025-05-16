from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def linear_regression_model(w, b, x):
    return np.dot(x, w) + b

def cost_function(w, b, x, m, target, lambda_):
    f_wb = np.dot(x, w) + b
    temp = f_wb - target
    J_wb = np.dot(temp, temp) / (2 * m)
    J_wb += (np.sum(w ** 2) * lambda_) / (2 * m)
    return J_wb

def compute_gradient(x, target, w, b, lambda_):
    m = x.shape[0]
    dj_dw = (np.dot(x.T, (np.dot(x, w) + b - target)) / m) + (lambda_ * w / m)
    dj_db = np.sum(np.dot(x, w) + b - target) / m
    return dj_dw, dj_db

def gradient_descent(iterations, alpha, x, target, w, b, lambda_):
    J_history = []
    for i in range(iterations):
        dj_dw, dj_db = compute_gradient(x, target, w, b, lambda_)
        
        w -= alpha * dj_dw
        b -= alpha * dj_db

        if i < 100000:
            J_history.append(cost_function(w, b, x, m, target, lambda_))
        """ if i % math.ceil(iterations / 10) == 0:
            formatted_w = ", ".join([f"{x:0.3e}" for x in w])
            print(f"Iteration {i:4}: \n Cost {J_history[-1]:0.2e} ", f"w: {formatted_w} \n b: {b: 0.5e}") """
    return w, b, J_history


# Load the diabetes dataset

diabetes = load_diabetes()
target = diabetes.target
X = np.array(diabetes.data)
""" X = np.delete(data, 1, axis=1)  # Remove one feature for simplicity """
features = diabetes.feature_names
print(f"X before scaling: {X}")


# Standardize the features

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"X after scaling: {X_scaled}")


X_train, X_test, y_train, y_test = train_test_split(X_scaled, target, test_size=0.3, random_state=42)


m = X_train.shape[0]
alpha = 7e-3
lambda_ = 12.5
iterations = 50000
w = np.zeros(X_train.shape[1])
b = 0.0


# Run gradient descent to find optimal w and b

w, b, J_cost = gradient_descent(iterations, alpha, X_train, y_train, w, b, lambda_)
print(f"W: {w} \n b: {b}")

# Make predictions on the scaled data

y_pred = linear_regression_model(w, b, X_test)

sns.scatterplot(x=y_pred, y=y_test)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2, linestyle='--', label='Ideal Line (y=x)')
plt.xlabel("Predictions")
plt.title("Evaluation of our LR model")
plt.grid(True)

# Evaluate the model using different metrics

mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
r2 = r2_score(y_test, y_pred)  # R-squared
mae = mean_absolute_error(y_test, y_pred)  # Mean Absolute Error
rmse = np.sqrt(mse)  # Root Mean Squared Error

# Print the evaluation results
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# Output the final parameters (w and b)
formatted_w = ", ".join([f"{x:8.4f}" for x in w])  
print(f"(w,b) found by gradient descent: ({formatted_w},{b:8.4f})")

plt.show()


