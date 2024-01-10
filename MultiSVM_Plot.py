#Divya Darshi
#1002090905

# CODE : Plotting part of MultiSVM Class

import matplotlib.pyplot as plt
from mlxtend.plotting.decision_regions import plot_decision_regions
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from MultiSVM import MultiSVM

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

 # Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

# Train the MultiSVM with a polynomial kernel
multi_svm_poly = MultiSVM(kernel='poly', C=1.0, degree=3)
# # Train the MultiSVM with a linear kernel
multi_svm_linear = MultiSVM(kernel='linear', C=1.0)

# Compare with scikit-learn's SVC
svm_poly = SVC(kernel='poly', C=1.0, degree=3)
svm_linear = SVC(kernel='linear', C=1.0)

# Create an instance of PCA for 2D reduction
pca = PCA(n_components=2)

# Reduce the data to 2D using PCA
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Select the petal length and petal width features for visualization
X_train_ = X_train[:, 2:4]
X_test_ = X_test[:, 2:4]

# Fit the Sklearn SVM with linear kernel on the reduced 2D data
svm_linear.fit(X_train_, y_train)
y_pred_sklearn_poly = svm_linear.predict(X_test_)

# Plot the decision boundary of the Sklearn with linear kernel
fig, ax = plt.subplots()
plot_decision_regions(X_test_, y_test, clf=svm_linear, ax=ax)
plt.title("Sklearn - linear Kernel")
plt.xlabel("feature 1")
plt.ylabel("feature 2")
plt.show()

# Fit the Sklearn SVM with poly kernel on the reduced 2D data
svm_poly.fit(X_train_, y_train)
y_pred_sklearn_poly = svm_poly.predict(X_test_)

# Plot the decision boundary of the Sklearn with poly kernel
fig, ax = plt.subplots()
plot_decision_regions(X_test_, y_test, clf=svm_poly, ax=ax)
plt.title("Sklearn - Poly Kernel")
plt.xlabel("feature 1")
plt.ylabel("feature 2")
plt.show()

# Fit the MultiSVM with Linear Kernel on the reduced 2D data
multi_svm_linear.fit(X_train_pca, y_train)
y_pred_multi_linear = multi_svm_linear.predict(X_test_pca)

# Plot the decision boundary for MultiSVM with Linear Kernel on the reduced data
fig, ax = plt.subplots()
plot_decision_regions(X_test_pca, y_test, clf=multi_svm_linear, ax=ax)
plt.title("MultiSVM - Linear Kernel")
plt.xlabel("feature 1")
plt.ylabel("feature 2")
plt.show()

# Fit the MultiSVM with Poly Kernel on the reduced 2D data
multi_svm_poly.fit(X_train_pca, y_train)
y_pred_multi_poly = multi_svm_poly.predict(X_test_pca)

# Plot the decision boundary for MultiSVM with Poly Kernel on the reduced data
fig, ax = plt.subplots()
plot_decision_regions(X_test_pca, y_test, clf=multi_svm_poly, ax=ax)
plt.title("MultiSVM - Poly Kernel")
plt.xlabel("feature 1")
plt.ylabel("feature 2")
plt.show()