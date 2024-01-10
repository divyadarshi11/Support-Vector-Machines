#Divya Darshi
#1002090905

import numpy as np

# Define the non-positive semidefinite kernel function
def non_psd_kernel(x1, x2):
    dot_product = np.dot(x1, x2)
    return dot_product**2

# Choose two input vectors
x1 = np.array([1, 2])
x2 = np.array([3, 4])

# Compute the kernel matrix
kernel_matrix = np.array([[non_psd_kernel(x1, x1), non_psd_kernel(x1, x2)],
                          [non_psd_kernel(x2, x1), non_psd_kernel(x2, x2)]])

# Check if the kernel matrix is positive semidefinite
eigenvalues, _ = np.linalg.eig(kernel_matrix)
is_positive_semidefinite = all(eigenvalues >= 0)

print("Kernel Matrix:", kernel_matrix)
print("Is Positive Semidefinite:", is_positive_semidefinite)


