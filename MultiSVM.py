#Divya Darshi
#1002090905

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.utils.extmath import safe_sparse_dot

# SVM Class with SMO Algorithm
class NewSVM:
    def __init__(self, kernel='linear', C=1.0, degree=3, tol=1e-3, max_iter=1000):
        self.kernel = kernel
        self.C = C
        self.degree = degree
        self.tol = tol
        self.max_iter = max_iter
        self.alpha = None
        self.support_vectors = None
        self.support_vector_labels = None

    def fit(self, X, y, class_label):
        n_samples, n_features = X.shape

        # Convert class labels to binary labels
        binary_y = np.where(y == class_label, 1, -1)

        # Initialize alpha and bias
        self.alpha = np.zeros(n_samples)
        bias = 0.0

        for _ in range(self.max_iter):
            num_changed_alphas = 0
            for i in range(n_samples):
                # Calculate the prediction for sample i
                prediction = bias + np.sum(self.alpha * binary_y * self.kernel_function(X, X[i]))

                # Compute the error
                error = prediction - binary_y[i]

                # Check KKT conditions
                if (binary_y[i] * error < -self.tol and self.alpha[i] < self.C) or \
                   (binary_y[i] * error > self.tol and self.alpha[i] > 0):

                    # Select a random j not equal to i
                    j = i
                    while j == i:
                        j = np.random.randint(0, n_samples)

                    # Calculate the prediction for sample j
                    prediction_j = bias + np.sum(self.alpha * binary_y * self.kernel_function(X, X[j]))

                    # Compute the error for sample j
                    error_j = prediction_j - binary_y[j]

                    # Save the old alpha values
                    alpha_i_old = self.alpha[i]
                    alpha_j_old = self.alpha[j]

                    # Compute the bounds for alpha[i] and alpha[j]
                    if binary_y[i] != binary_y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])

                    if L == H:
                        continue

                    # Compute the kernel similarity and eta
                    eta = 2.0 * self.kernel_function(X[i], X[j]) - self.kernel_function(X[i], X[i]) - self.kernel_function(X[j], X[j])
                    if eta >= 0:
                        continue

                    # Update alpha[j]
                    self.alpha[j] = alpha_j_old - binary_y[j] * (error - error_j) / eta

                    # Clip alpha[j] to be within [L, H]
                    self.alpha[j] = min(H, max(L, self.alpha[j]))

                    # Check if alpha[j] has changed significantly
                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue

                    # Update alpha[i] using the same step in the opposite direction
                    self.alpha[i] = alpha_i_old + binary_y[i] * binary_y[j] * (alpha_j_old - self.alpha[j])

                    # Compute the bias
                    b1 = bias - error - binary_y[i] * (self.alpha[i] - alpha_i_old) * self.kernel_function(X[i], X[i]) \
                         - binary_y[j] * (self.alpha[j] - alpha_j_old) * self.kernel_function(X[i], X[j])
                    b2 = bias - error_j - binary_y[i] * (self.alpha[i] - alpha_i_old) * self.kernel_function(X[i], X[j]) \
                         - binary_y[j] * (self.alpha[j] - alpha_j_old) * self.kernel_function(X[j], X[j])
                    if 0 < self.alpha[i] < self.C:
                        bias = b1
                    elif 0 < self.alpha[j] < self.C:
                        bias = b2
                    else:
                        bias = (b1 + b2) / 2.0

                    num_changed_alphas += 1

            if num_changed_alphas == 0:
                break

        # Select support vectors and their labels
        support_vector_indices = np.where(self.alpha > 0)[0]
        self.support_vectors = X[support_vector_indices]
        self.support_vector_labels = binary_y[support_vector_indices]

    def predict(self, X):
        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples)

        for i in range(n_samples):
            prediction = 0
            for j in range(len(self.support_vectors)):
                prediction += self.alpha[j] * self.support_vector_labels[j] * self.kernel_function(self.support_vectors[j], X[i])

            y_pred[i] = prediction

        return np.sign(y_pred)

    def kernel_function(self, X1, X2):
        if self.kernel == 'linear':
            return safe_sparse_dot(X1, X2.T)
        elif self.kernel == 'poly':
            return (safe_sparse_dot(X1, X2.T) + 1) ** self.degree

# MultiClassSVM
class MultiSVM:
    def __init__(self, kernel='linear', C=1.0, degree=3):
        self.kernel = kernel
        self.C = C
        self.degree = degree
        self.svm_classifiers = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        for cls in self.classes:
            svm_classifier = NewSVM(kernel=self.kernel, C=self.C, degree=self.degree)
            svm_classifier.fit(X, y, cls)
            self.svm_classifiers[cls] = svm_classifier

    def predict(self, X):
        n_samples = X.shape[0]
        predictions = np.zeros((n_samples, len(self.classes)))

        for i in range(len(self.classes)):
            predictions[:, i] = self.svm_classifiers[self.classes[i]].predict(X)

        return np.array([self.classes[np.argmax(pred)] for pred in predictions])

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the MultiSVM with a linear kernel
multi_svm_linear = MultiSVM(kernel='linear', C=1.0)
multi_svm_linear.fit(X_train, y_train)
y_pred_linear = multi_svm_linear.predict(X_test)

# Train the MultiSVM with a polynomial kernel
multi_svm_poly = MultiSVM(kernel='poly', C=1.0, degree=3)
multi_svm_poly.fit(X_train, y_train)
y_pred_poly = multi_svm_poly.predict(X_test)


# Evaluate the models
accuracy_linear = accuracy_score(y_test, y_pred_linear)
accuracy_poly = accuracy_score(y_test, y_pred_poly)

print("Accuracy (Multi SVM with Linear Kernel):", accuracy_linear)
print("Accuracy ( Multi SVM with Poly Kernel):", accuracy_poly)

# Compare with scikit-learn's SVC
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_train, y_train)
y_pred_sklearn_linear = svm_linear.predict(X_test)

svm_poly = SVC(kernel='poly', C=1.0, degree=3)
svm_poly.fit(X_train, y_train)
y_pred_sklearn_poly = svm_poly.predict(X_test)

accuracy_sklearn_linear = accuracy_score(y_test, y_pred_sklearn_linear)
accuracy_sklearn_poly = accuracy_score(y_test, y_pred_sklearn_poly)

print("Accuracy (sklearn SVM with Linear Kernel):", accuracy_sklearn_linear)
print("Accuracy (sklearn SVM with Poly Kernel):", accuracy_sklearn_poly)

