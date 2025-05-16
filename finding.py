import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the dataset
data = load_diabetes()
X = data.data[:, 0].reshape(-1, 1)  # Example feature
y = data.target

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Testing different polynomial degrees
degrees = range(1, 10)  # Trying 1st to 5th degree polynomials
errors = []

for d in degrees:
    poly = PolynomialFeatures(degree=d)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_poly_train, y_train)
    y_pred = model.predict(X_poly_test)
    
    mse = mean_squared_error(y_test, y_pred)
    errors.append(mse)

# Plot the results
plt.plot(degrees, errors, marker='o')
plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error")
plt.title("Finding the Best Polynomial Degree")
plt.show()