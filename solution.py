# First Task :
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# A Function to Generate Data 
def toydata_generate(f, sizeofthesample, varation_of_the_noise):
    x = np.linspace(0, 1, sizeofthesample)
    t = f(x) + np.random.normal(scale=varation_of_the_noise, size=x.shape)
    return x, t

def f(x):
    return np.sin(2 * np.pi * x)

#1. Generate 10 points for training, 100 points for testing
x_train, y_train = toydata_generate(f, 10, 0.25)
x_test = np.linspace(0, 1, 100)
y_test = f(x_test)

#2. Apply polynomial basis function (order M=9)
poly = PolynomialFeatures(degree=9)
Phi_train = poly.fit_transform(x_train[:, np.newaxis])
Phi_test = poly.transform(x_test[:, np.newaxis])

# 3. Train model in parametric way and report test MSE
lr = LinearRegression(fit_intercept=False)
lr.fit(Phi_train, y_train)
y_pred = lr.predict(Phi_test)
mse = mean_squared_error(y_test, y_pred)
print("Test MSE (parametric):", mse)

# 4. Get prediction in non-parametric way
K = Phi_train @ Phi_train.T
k = Phi_test @ Phi_train.T
y_pred_non_param = k @ np.linalg.inv(K) @ y_train
mse_non_param = mean_squared_error(y_test, y_pred_non_param)
print("Test MSE (non-parametric):", mse_non_param)

# 5. Compare the predictions
print("Are the predictions identical?", np.allclose(y_pred, y_pred_non_param))
# ```

# Task 2:
# ```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import mean_squared_error

#1.  Define gram matrix using the RBF kernel
gamma = 5
K = rbf_kernel(X=x_train[:, np.newaxis], Y=x_train[:, np.newaxis], gamma=gamma)

# Covariance matrix, beta = 10
beta = 10
C = K + np.eye(len(K)) / beta

#2.  Check whether C is invertible or not
try:
    C_inv = np.linalg.inv(C)
    print("C is invertible.")
except np.linalg.LinAlgError:
    print("C is not invertible.")

#3. Compute the predictive mean for all test samples
k_test = rbf_kernel(X=x_test[:, np.newaxis], Y=x_train[:, np.newaxis], gamma=gamma)
y_pred_GP = k_test @ C_inv @ y_train

# Test MSE
mse_GP = mean_squared_error(y_test, y_pred_GP)
print("Test MSE (Gaussian Process):", mse_GP)
# ```

# Task 3:
# ```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load iris dataset and split into training and testing
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=5)

# Create a SVC model and train it
svc = SVC()
svc.fit(X_train, y_train)

# 2.1: SVC handles multi-class classification using one-vs-one approach by default
print("SVC uses one-vs-one approach for multi-class classification")

# 2.2: Number of support vectors
print("Number of support vectors:", len(svc.support_))

# # 2Continuing from Task 3:

# ```python
# 2.3: Check whether the 18th training sample is a support vector
is_support_vector = 18 in svc.support_
print("Is the 18th training sample a support vector?", is_support_vector)

# 2.4: Number of support vectors from class 2
n_support_vectors_class_2 = svc.n_support_[2]
print("Number of support vectors from class 2:", n_support_vectors_class_2)

# 3: Report the classification test accuracy
y_pred = svc.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print("Classification test accuracy:", test_accuracy)
# ```

# This concludes Task 3. The code above checks if the 18th training sample is a support vector, calculates the number of support vectors from class 2, and reports the classification test accuracy for the SVM model.
#TASK 4 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_squared_error

def acquisition_function(mean, std, epsilon=0.01):
    return mean / (std + epsilon)

# Load data
x = np.load('BO_x.npy')
y = np.load('BO_y.npy')

# Randomly select 10 samples as initial training data
indices = np.random.choice(len(x), 10, replace=False)
x_train = x[indices]
y_train = y[indices]

# Treat the rest of the samples as candidate samples
candidate_indices = np.array(list(set(range(len(x))) - set(indices)))
candidate_x = x[candidate_indices]

# Bayesian optimization
num_iterations = 500
bayesian_optimization_curve = []
random_curve = []

for i in range(num_iterations):
    # Train GP model
    kernel = RBF(length_scale=1.0)
    gp = GaussianProcessRegressor(kernel=kernel)
    gp.fit(x_train, y_train)

    # Predict mean and std for candidate samples
    mean, std = gp.predict(candidate_x, return_std=True)

    # Select next sample using acquisition function
    acquisition_values = acquisition_function(mean, std)
    max_index = np.argmax(acquisition_values)
    next_x = candidate_x[max_index]
    next_y = y[candidate_indices[max_index]]

    # Add selected sample to training data
    x_train = np.vstack([x_train, next_x])
    # y_train = np.vstack([y_train, next_y])
    # y_train = np.vstack([y_train, next_y.reshape(1, 1)])
    y_train = np.concatenate([y_train, np.array([next_y])])



    # Remove selected sample from candidate pool
    candidate_x = np.delete(candidate_x, max_index, axis=0)
    candidate_indices = np.delete(candidate_indices, max_index)

    # Calculate and save f(x) for Bayesian optimization curve
    bayesian_optimization_curve.append(next_y)

    # Random selection for random curve
    random_index = np.random.choice(len(candidate_x))
    random_y = y[candidate_indices[random_index]]
    random_curve.append(random_y)

# Calculate cumulative epsilon-optimal sample curve
y_min = np.min(y)
epsilon = 0.5

bayesian_optimal_samples = np.cumsum(np.abs(np.array(bayesian_optimization_curve) - y_min) / y_min < epsilon)
random_optimal_samples = np.cumsum(np.abs(np.array(random_curve) - y_min) / y_min < epsilon)

# Plot results
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(bayesian_optimization_curve, label='Bayesian Optimization')
plt.plot(random_curve, label='Random')
plt.xlabel('Iteration')
plt.ylabel('f(x)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(bayesian_optimal_samples, label='Bayesian Optimization')
plt.plot(random_optimal_samples, label='Random')
plt.xlabel('Iteration')
plt.ylabel('Cumulative Epsilon-Optimal Samples')
plt.legend()

plt.show()


