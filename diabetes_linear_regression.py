from sklearn.datasets import load_diabetes 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

def linear_regression_model(w, b, x):
    return np.dot(x, w) + b

def cost_function(w, b, x, m, target, lambda_) :
    J_wb = 0
    temp = 0
    """ for i in range(m):
        f_wb = w * x[i] + b
        temp = temp + (f_wb - target[i])**2
    J_wb = temp/(2*m) """
    f_wb = np.dot(x, w) + b
    temp = f_wb - target
    J_wb = np.dot(temp,temp) / (2 * m)
    J_wb += (np.sum(w ** 2) * lambda_) / (2 * m)
    return J_wb

def compute_gradient(x, target, w, b, lambda_):
    m = x.shape[0]
    """ dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - target[i]) * x[i]
        dj_db_i = (f_wb - target[i]) 
        dj_dw += dj_dw_i
        dj_db += dj_db_i
    f_wb = np.dot(x, w) + b
    temp = np.sum(f_wb - target)
    dj_dw = temp
    dj_db = temp
    for i in range(len(w)):
        dj_dw += lambda_ * w[i]
    dj_dw /= m
    dj_db /= m """
    dj_dw = (np.dot(X.T, (np.dot(X, w) + b - target)) / m) + (lambda_ * w / m)
    dj_db = np.sum(np.dot(X, w) + b - target) / m
    return dj_dw, dj_db

def gradient_descent(iterations, alpha, x, target, w, b, lambda_):
    J_history = []
    p_history = []
    for i in range(iterations) :
        dj_dw, dj_db = compute_gradient(x, target, w, b, lambda_)
        
        w -= alpha * dj_dw
        b -= alpha * dj_db

        if i<100000:
            J_history.append(cost_function(w, b, x, m, target, lambda_))
            p_history.append([w,b])
        if i%math.ceil(iterations/10) == 0:
            formatted_dj_dw = ", ".join([f"{x:0.3e}" for x in dj_dw])  
            """  formatted_dj_db = ", ".join([f"{x:0.3e}" for x in dj_db])  """
            formatted_w = ", ".join([f"{x:0.3e}" for x in w])
            print(f"Iteration {i:4}: \n Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {formatted_dj_dw}, dj_db: {dj_db : 0.3e}  \n",
                  f"w: {formatted_w} \n b:{b: 0.5e}")
    return w, b, J_history, p_history    
            

diabetes = load_diabetes()
target = diabetes.target
data = np.array(diabetes.data.data)
X = np.delete(data, 1, axis=1)
features = diabetes.feature_names

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

""" plt.figure(figsize=(15,10))
for i in range(len(features)):
    plt.subplot(3, 4, i+1)
    plt.scatter(X[ : , i], target, c = 'red', s=5)
    plt.title(features[i])
    plt.xlabel(features[i])
    plt.ylabel("Target")
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4, hspace=0.6)


plt.suptitle("Features v/s Target")
plt.show() """



m = X.shape[0]
alpha = 1e-3
lambda_ = 1.3
iterations = 50000
w = np.zeros(X.shape[1])
b = 0.0
w, b, J_cost, p_history = gradient_descent(iterations, alpha, x= X_scaled, target = diabetes.target, w= w ,b= b, lambda_ = lambda_)
formatted_w = ", ".join([f"{x:8.4f}" for x in w])  
print(f"(w,b) found by gradient descent: ({formatted_w},{b:8.4f})")


n = X.shape[1]  # Number of features
fig, axes = plt.subplots(5, 2, figsize=(15, 10))  # Create subplots for each feature

for i, ax in enumerate(axes.flat):
    if i < n:
        # Step 1: Generate linearly spaced values for the current feature, in the scaled space
        x_vals = np.linspace(X_scaled[:, i].min(), X_scaled[:, i].max(), 442)

        # Step 2: Create a temporary feature matrix where only the i-th feature varies
        X_temp = np.zeros((442, X.shape[1]))  # All other features are set to 0 (mean)
        X_temp[:, i] = x_vals  # The i-th feature varies

        # Step 3: Predict the target using the model (using the scaled data)
        y_vals = linear_regression_model(w, b, X_temp)

        # Step 4: Plot the actual data and predicted line
        ax.scatter(X_scaled[:, i], target, color="red", s=5, label="Actual")  # Scatter plot for actual data
        ax.plot(x_vals, y_vals, color="blue", label="Predicted")  # Regression line using x_vals

        ax.set_title(features[i])
        ax.set_xlabel(features[i])
        ax.set_ylabel("Diabetes Progression")
        ax.legend()

plt.tight_layout()  # Adjust the layout
plt.suptitle("Feature-wise Linear Regression Results")  # Add a title
plt.show()
