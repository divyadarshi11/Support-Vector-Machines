#Divya Darshi
#1002090905

import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting.decision_regions import plot_decision_regions
from sklearn.svm import LinearSVC

#Define Linear kernel function
def linear_kernel(x1, x2):
    return x1.T @ x2

class SVM:
    def __init__(self, kernel='linear', c=1.0, tol=1e-3, maxiter=1000):
        self._kernel = kernel
        self._tol = tol
        self._maxiter = maxiter
        
        if self._kernel == 'linear':
            self._k = linear_kernel
        
        self._c = c
        
    def _init_params(self):
        self._error_cache = np.zeros(self._data.shape[0])
        self._alphas = np.ones(self._data.shape[0]) * .1
        self._b = 0
        
        if self._kernel == 'linear':
            self._weights = np.random.rand(self._data.shape[1])

    def predict_score(self, x):
        u = 0
        if self._kernel == 'linear':
            u = self._weights @ x.T - self._b
        else:
            for i in range(self._data.shape[0]):
                u += self._targets[i] * self._alphas[i] * self._k(self._data[i], x)
            u -= self._b

        return u
        
    def predict(self, x):
        score = self.predict_score(x)

        if type(score) is np.ndarray:
            score[score < 0] = -1
            score[score >= 0] = 1

            return score
        else:
            return -1 if score < 0 else 1

    def smo_step(self, i1, i2):
        if i1 == i2:
            return 0

        x1 = self._data[i1]
        x2 = self._data[i2]
        y1 = self._targets[i1]
        y2 = self._targets[i2]
        alpha1 = self._alphas[i1]
        alpha2 = self._alphas[i2]

        # Compute errors for x1 and x2
        e1 = self.predict_score(x1) - y1
        e2 = self.predict_score(x2) - y2

        s = y1 * y2

        if s == 1:
            L = max(0, alpha2 + alpha1 - self._c)
            H = min(self._c, alpha2 + alpha1)
        else:
            L = max(0, alpha2 - alpha1)
            H = min(self._c, self._c + alpha2 - alpha1)

        if L == H:
            return 0

        k11 = self._k(x1, x1)
        k22 = self._k(x2, x2)
        k12 = self._k(x1, x2)

        eta = k11 + k22 - 2 * k12

        if eta > 0:
            a2 = alpha2 + y2 * (e1 - e2) / eta
            if a2 <= L:
                a2 = L
            elif a2 >= H:
                a2 = H
        else:
            # Negative case: Calculate objective function values at a2 = L and a2 = H
            f1 = y1 * (e1 + self._b) - alpha1 * k11 - alpha2 * k12
            f2 = y2 * (e2 + self._b) - alpha1 * k12 - alpha2 * k22
            Lobj = alpha1 + s * (alpha2 - L) - 0.5 * k11 * (alpha1 - L) ** 2 - 0.5 * k22 * (alpha2 - L) ** 2 - s * k12 * (alpha1 - L) * (alpha2 - L) - f1 - f2
            Hobj = alpha1 + s * (alpha2 - H) - 0.5 * k11 * (alpha1 - H) ** 2 - 0.5 * k22 * (alpha2 - H) ** 2 - s * k12 * (alpha1 - H) * (alpha2 - H) - f1 - f2

            if Lobj < Hobj - 1e-3:
                a2 = L
            elif Lobj > Hobj + 1e-3:
                a2 = H
            else:
                a2 = alpha2
 
        if np.abs(a2 - alpha2) < 1e-3 * (a2 + alpha2 + 1e-3):
            return 0
  
        a1 = alpha1 + s * (alpha2 - a2)

        # Update threshold to reflect change in Lagrange multipliers
        b1 = e1 + y1 * (a1 - alpha1) * k11 + y2 * (a2 - alpha2) * k12 + self._b
        b2 = e2 + y1 * (a1 - alpha1) * k12 + y2 * (a2 - alpha2) * k22 + self._b
        self._b = (b1 + b2) / 2

        # Update weight vector to reflect change in a1 & a2, if SVM is linear
        if self._kernel == 'linear':
            self._weights = np.sum((self._targets * self._alphas)[:, None] * self._data, axis=0)
        
        # Store a1 and a2 in the alpha array
        self._alphas[i1] = a1
        self._alphas[i2] = a2

        # Update error cache using new multipliers
        for i in range(self._data.shape[0]):
            self._error_cache[i] = self.predict_score(self._data[i]) - self._targets[i]

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

    def fit(self, data, targets):
        self._data = data
        self._targets = targets
        
        self._init_params()
        
        n_changed = 0
        examine_all = True
        n_iter = 0
        
        while (n_changed > 0 or examine_all is True) and n_iter < self._maxiter:
            n_changed = 0
            n_iter += 1
            
            if examine_all is True:
                random_idxs = np.random.permutation(np.arange(data.shape[0]))
                for i in random_idxs:
                    n_changed += self.examine(i)
            else:
                f_idxs = np.where((self._alphas != 0) & (self._alphas != self._c))[0]
                random_idxs = np.random.permutation(f_idxs)
                for i, v in enumerate(random_idxs):
                    n_changed += self.examine(v)
            
            if examine_all is True:
                examine_all = False
            elif n_changed == 0:
                examine_all = True
                
                
def main():
    # Generate sample data
    n_a_samples = 50
    n_b_samples = 50
    a_samples = np.random.multivariate_normal([-1, 1], [[0.1, 0], [0, 0.1]], n_a_samples)
    b_samples = np.random.multivariate_normal([1, -1], [[0.1, 0], [0, 0.1]], n_b_samples)
    a_targets = np.ones(n_a_samples) * -1  
    b_targets = np.ones(n_b_samples)  
    samples = np.concatenate((a_samples, b_samples))
    targets = np.concatenate((a_targets, b_targets))
    print(samples.shape, targets.shape)

    # Create an instance of SVM class
    custom_svm = SVM(c=1.0)

    # Fit the custom SVM model to the data
    custom_svm.fit(samples, targets)

    print(f"Weights: {custom_svm._weights}")
    print(f"b: {custom_svm._b}")

    # Create and fit the LinearSVC model
    linear_svc = LinearSVC(dual=False)
    linear_svc.fit(samples, targets.astype(np.int32))

    # Display LinearSVC model parameters
    print(f"coef_: {linear_svc.coef_}")
    print(f"intercept: {linear_svc.intercept_}")

    # Plot decision regions for custom SVM
    fig = plt.figure()
    ax = plot_decision_regions(samples, targets.astype(np.int32), custom_svm)
    fig.add_subplot(ax)
    plt.title("Custom SVM Decision Region")
    plt.show()    

    # Plot decision boundary for LinearSVC
    plt.figure()
    plot_decision_regions(samples, targets.astype(np.int32), linear_svc)
    plt.title("LinearSVC Decision Region")
    plt.show()

if __name__ == "__main__":
    main()



