#Divya Darshi
#1002090905

import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting.decision_regions import plot_decision_regions
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# Define the linear kernal 
def linear_kernel(x1, x2):
    return x1.T @ x2

# Define polynomial kernel
def polynomial_kernel(x1, x2, degree=3):
    return (np.dot(x1, x2) + 1) ** degree

class Non_linearSVM():
    def __init__(self, kernel='poly', C=1.0, tol=1e-3, maxiter=1000, degree=3):
        self.kernel = kernel
        self.C = C
        self.tol = tol
        self.maxiter = maxiter
        self.degree = degree
        self.alphas = None
        self.b = None

    def _init_params(self, X):
        self.n_samples, self.n_features = X.shape
        self.error_cache = np.zeros(self.n_samples)
        self.alphas = np.zeros(self.n_samples)
        self.b = 0

    def predict_score(self, X):
        if self.kernel == 'linear':
            return np.dot(X, self.weights) - self.b
        elif self.kernel == 'poly':
            scores = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                score = -self.b
                for j in range(self.n_samples):
                    score += self.alphas[j] * self.y[j] * polynomial_kernel(X[i], self.X[j], self.degree) 
                scores[i] = score
            return scores
        
    def predict(self, X):
        scores = self.predict_score(X)
        return np.where(scores >= 0, 1, -1)

    def _smo_step(self, i, j):
        if i == j:
            return 0

        alpha_i, alpha_j = self.alphas[i], self.alphas[j]
        x_i, x_j, y_i, y_j = self.X[i], self.X[j], self.y[i], self.y[j]
        E_i, E_j = self.error_cache[i], self.error_cache[j]

        if y_i != y_j:
            L = max(0, alpha_j - alpha_i)
            H = min(self.C, self.C + alpha_j - alpha_i)
        else:
            L = max(0, alpha_i + alpha_j - self.C)
            H = min(self.C, alpha_i + alpha_j)

        if L == H:
            return 0

        eta = 2 * polynomial_kernel(x_i, x_j, self.degree) - \
              polynomial_kernel(x_i, x_i, self.degree) - \
              polynomial_kernel(x_j, x_j, self.degree)
        if eta >= 0:
            return 0

        alpha_j_new = alpha_j - (y_j * (E_i - E_j)) / eta
        alpha_j_new = max(L, min(H, alpha_j_new))

        if abs(alpha_j - alpha_j_new) < 1e-5:
            return 0

        alpha_i_new = alpha_i + y_i * y_j * (alpha_j - alpha_j_new)

        self.b = self._compute_b(E_i, E_j, alpha_i, alpha_j, alpha_i_new, alpha_j_new, i, j)

        self.alphas[i] = alpha_i_new
        self.alphas[j] = alpha_j_new

        self.error_cache[i] = self._compute_error(x_i, y_i, self.alphas[i]) - y_i
        self.error_cache[j] = self._compute_error(x_j, y_j, self.alphas[j]) - y_j

        return 1
    
    def examine(self, i2):
        x2 = self._data[i2]
        y2 = self._targets[i2]
        alpha2 = self._alphas[i2]
        e2 = self.predict_score(x2) - y2
        r2 = e2 * y2

        if (r2 < -self._tol and alpha2 < self._c) or (r2 > self._tol and alpha2 > 0):
            f_idxs = np.where((self._alphas != 0) & (self._alphas != self._c))[0]

            if len(f_idxs) > 1:
                max_step = 0
                for i, v in enumerate(f_idxs):
                    if v == i2:
                        continue

                    if self._error_cache[v] == 0:
                        self._error_cache[v] = self.predict_score(self._data[v]) - self._targets[v]
                    step = np.abs(self._error_cache[v] - e2)

                    if step > max_step:
                        max_step = step
                        i1 = v

                if self.smo_step(i1, i2):
                    return 1

                for i, v in enumerate(np.random.permutation(f_idxs)):
                    if self.smo_step(v, i2):
                        return 1

                for i, v in enumerate(np.random.permutation(range(self._data.shape[0]))):
                    if v == i2:
                        continue
                    if self.smo_step(v, i2):
                        return 1

        return 0

    
    def fit(self, X, y):
        self.X = X
        self.y = y
        self._init_params(X)

        for _ in range(self.maxiter):
            num_changed_alphas = 0
            for i in range(self.n_samples):
                if self.error_cache[i] * self.y[i] < -self.tol and self.alphas[i] < self.C \
                        or self.error_cache[i] * self.y[i] > self.tol and self.alphas[i] > 0:
                    j = np.random.choice(np.arange(self.n_samples)[i != np.arange(self.n_samples)])
                    num_changed_alphas += self._smo_step(i, j)
            if num_changed_alphas == 0:
                break

        self.support_vectors = np.where(self.alphas > 0)[0]
        if self.kernel == 'linear':
            self.weights = np.sum(self.alphas[self.support_vectors] * y[self.support_vectors]
                                  * X[self.support_vectors].T, axis=1)
        elif self.kernel == 'poly':
            self.weights = None

# Generate a non-linear dataset
X, y = make_circles(n_samples=500, factor=0.3, noise=0.05, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

# Use a smaller subset of data for visualization 
X_subset = X_train[:100]
y_subset = y_train[:100]

# Test SVM with a polynomial kernel
Non_linear_svm = Non_linearSVM(kernel='poly', degree=3, C=1)
Non_linear_svm.fit(X_subset, y_subset)

# Print the accuracy of the implementation on the test set
y_pred_custom = Non_linear_svm.predict(X_test)
accuracy_custom = accuracy_score(y_test, y_pred_custom)
print(f"Accuracy (Non linear SVM - Polynomial Kernel): {accuracy_custom:.2f}")

# Compare with scikit-learn's SVC
svc_linear = SVC(kernel='linear', C=1)
svc_linear.fit(X_subset, y_subset)
y_pred_svc_linear = svc_linear.predict(X_test)
accuracy_svc_linear = accuracy_score(y_test, y_pred_svc_linear)

svc_poly = SVC(kernel='poly', degree=3, C=1)
svc_poly.fit(X_subset, y_subset)
y_pred_svc_poly = svc_poly.predict(X_test)
accuracy_svc_poly = accuracy_score(y_test, y_pred_svc_poly)

print(f"Accuracy (sklearn SVC - Linear Kernel): {accuracy_svc_linear:.2f}")
print(f"Accuracy (sklearn SVC - Polynomial Kernel): {accuracy_svc_poly:.2f}")

# Plot the decision boundary of the Sklearn with linear kernel
fig, ax = plt.subplots()
plot_decision_regions(X_subset, y_subset, clf=svc_linear, ax=ax)
plt.title("Sklearn - linear Kernel")
plt.show()

# Plot the decision boundary of the SKlearn with  polynomial kernel
fig, ax = plt.subplots()
plot_decision_regions(X_subset, y_subset, clf=svc_poly, ax=ax)
plt.title("Sklearn - Polynomial Kernel")
plt.show()

# Plot the decision boundary of the Non_linear SVM with a polynomial kernel
fig, ax = plt.subplots()
plot_decision_regions(X_subset, y_subset, clf=Non_linear_svm, ax=ax)
plt.title("Non Linear SVM - Polynomial Kernel")
plt.show()

