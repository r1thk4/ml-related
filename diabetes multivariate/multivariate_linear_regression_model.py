from sklearn.datasets import load_diabetes 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import math

# Linear regression model for prediction
def linear_regression_model(w, b, x):
    return np.dot(x, w) + b

# Cost function with regularization
def cost_function(w, b, x, m, target, lambda_):
    f_wb = np.dot(x, w) + b  
    temp = f_wb - target
    J_wb = np.dot(temp, temp) / (2 * m)  
    J_wb += (np.sum(w ** 2) * lambda_) / (2 * m) 
    return J_wb

# Compute gradient of cost function with respect to weights and bias
def compute_gradient(x, target, w, b, lambda_):
    m = x.shape[0]
    dj_dw = (np.dot(x.T, (np.dot(x, w) + b - target)) / m) + (lambda_ * w / m)  
    dj_db = np.sum(np.dot(x, w) + b - target) / m  
    return dj_dw, dj_db

# Gradient descent function to update weights and bias
def gradient_descent(iterations, alpha, x, target, w, b, lambda_):
    J_history = []  # To track cost history
    p_history = []  # To track weights and bias history
    for i in range(iterations):
        dj_dw, dj_db = compute_gradient(x, target, w, b, lambda_)
        
        w -= alpha * dj_dw  # Update weights
        b -= alpha * dj_db  # Update bias

        if i < 100000:  # Store cost and parameters for plotting
            J_history.append(cost_function(w, b, x, m, target, lambda_))
            p_history.append([w, b])

        if i % math.ceil(iterations / 10) == 0:  # Print progress every 10% of iterations
            formatted_dj_dw = ", ".join([f"{x:0.3e}" for x in dj_dw])  
            formatted_w = ", ".join([f"{x:0.3e}" for x in w])
            print(f"Iteration {i:4}: \n Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {formatted_dj_dw}, dj_db: {dj_db : 0.3e}  \n",
                  f"w: {formatted_w} \n b:{b: 0.5e}")
    return w, b, J_history, p_history

# Load dataset and preprocess
diabetes = load_diabetes()
target = diabetes.target
data = np.array(diabetes.data.data)
X = np.delete(data, 1, axis=1)  # Remove second feature for simplicity
features = diabetes.feature_names

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize parameters
m = X.shape[0]  # Number of training examples
alpha = 7e-3  # Learning rate
lambda_ = 12.5  # Regularization strength
iterations = 50000  # Number of iterations for gradient descent
w = np.zeros(X.shape[1])  # Initialize weights as zeros
b = 0.0  # Initialize bias

# Run gradient descent to optimize w and b
w, b, J_cost, p_history = gradient_descent(iterations, alpha, x=X_scaled, target=diabetes.target, w=w, b=b, lambda_=lambda_)

# Print the results
formatted_w = ", ".join([f"{x:8.4f}" for x in w])  
print(f"(w,b) found by gradient descent: ({formatted_w},{b:8.4f})")

# Plot the results for each feature
n = X.shape[1]  # Number of features
fig, axes = plt.subplots(3, 3, figsize=(15, 10))  # Create subplots for each feature
fig.tight_layout(pad=1)
for i, ax in enumerate(axes.flat):
    if i < n:
        x_vals = np.linspace(X_scaled[:, i].min(), X_scaled[:, i].max(), 442)
        X_temp = np.zeros((442, X.shape[1]))  
        X_temp[:, i] = x_vals 

        y_vals = linear_regression_model(w, b, X_temp)

        ax.scatter(X_scaled[:, i], target, color="red", s=5, label="Actual")  # Scatter plot for actual data
        ax.plot(x_vals, y_vals, color="blue", label="Predicted")  # Regression line using x_vals

        ax.set_title(features[i])
        ax.set_xlabel(features[i])
        ax.set_ylabel("Diabetes Progression")
        ax.legend()

plt.suptitle("Feature-wise Linear Regression Results")  
plt.show()
