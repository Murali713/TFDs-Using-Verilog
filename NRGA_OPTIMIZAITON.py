import numpy as np

# Define the function psi(t) and its gradient and Hessian
def psi(t):
    t0, t1 = t
    return t0**2 + 2 * t1**2 + 3 * t0 * t1 - 4 * t0 - 5 * t1 + 6

def grad_psi(t):
    t0, t1 = t
    gradient_tk = np.array([2*t0 + 3*t1 - 4, 4*t1 + 3*t0 - 5])
    return gradient_tk

def hessian_psi(t):
    hessian_tk = np.array([[2, 3], [3, 4]])
    return hessian_tk

# NRGA optimization method
def nrga_optimization(initial_t, alpha, num_segments, max_iterations, tolerance):
    # Step 1: Initialize the parameter vector
    t_k = initial_t

    for iteration in range(1, max_iterations + 1):
        # Step 2: Compute the gradient
        gradient_tk = grad_psi(t_k)

        # Step 3: Calculate the Hessian matrix
        hessian_tk = hessian_psi(t_k)

        # Step 4: Solve for the update direction using Newton-Raphson method
        update_direction = np.linalg.solve(hessian_tk, gradient_tk)

        # Step 5: Find t_k+1 using the update equation
        t_k_1 = t_k - alpha * update_direction

        # Step 6: Evaluate the quadratic approximation
        quadratic_approximation = psi(t_k) - 0.5 * np.dot(gradient_tk, np.linalg.solve(hessian_tk, gradient_tk))

        # Step 7: Update the total gradient vector using update rule
        total_gradient = np.zeros_like(t_k)
        for i in range(num_segments):
            total_gradient += 1 / alpha * np.linalg.solve(hessian_tk, gradient_tk)

        # Step 8: Check for convergence
        if np.linalg.norm(t_k_1 - t_k) < tolerance:
            print(f"Convergence reached after {iteration} iterations.")
            break

        # Step 9: Update t_k for the next iteration
        t_k = t_k_1

    return t_k

# Example usage:
initial_t = np.array([0.0, 0.0])  # Replace with your initial parameter vector
alpha = 0.1  # Replace with the desired alpha value
num_segments = 10  # Replace with the number of segments
max_iterations = 100  # Replace with the desired maximum number of iterations
tolerance = 1e-6  # Replace with the desired tolerance level

resulting_t = nrga_optimization(initial_t, alpha, num_segments, max_iterations, tolerance)
print("Optimized parameter vector:", resulting_t)
