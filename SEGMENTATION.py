import numpy as np
from scipy.special import comb

def bernstein_poly(i, n, t):
    """
    Calculate the value of the Bernstein polynomial basis function b_i,n(t).
    """
    return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))

def calculate_bezier_curve(control_points, t):
    """
    Calculate the Bezier curve at parameter t using the given control points.
    """
    n = len(control_points) - 1
    bezier_curve = np.zeros_like(control_points[0])
    for i in range(n + 1):
        bezier_curve += control_points[i] * bernstein_poly(i, n, t)
    return bezier_curve

def nss_approximation(original_signal, d, tolerance):
    """
    Approximate the NSS using a Fraction Bezier Bernstein curve.

    Parameters:
        original_signal (numpy array): The original signal data points.
        d (int): Fractional degree of the Bezier curve.
        tolerance (float): Desired tolerance level for stopping the iteration.

    Returns:
        numpy array: The approximate signal obtained from the Bezier curve.
    """
    n = len(original_signal)
    control_points = original_signal.copy()

    while True:
        # Step 2: Calculate the Bernstein polynomial basis functions.
        t_range = np.linspace(0, 1, n)
        basis_functions = np.array([bernstein_poly(i, d, t_range) for i in range(d)])

        # Step 3: Construct the Bezier curve.
        bezier_curve = np.zeros_like(original_signal)
        for i in range(d):
            bezier_curve += np.outer(control_points[i], basis_functions[i])

        # Step 4: Identify points of inflection.
        second_derivative = np.gradient(np.gradient(bezier_curve, axis=0), axis=0)
        inflection_points = [t_range[i] for i in range(1, n - 1) if second_derivative[i - 1].dot(second_derivative[i]) < 0]

        # Step 5: Use inflection points as new control points.
        control_points = np.array([calculate_bezier_curve(control_points, t) for t in inflection_points])

        # Step 10: Calculate error and check convergence.
        approximated_signal = calculate_bezier_curve(control_points, t_range)
        error = np.sum((original_signal - approximated_signal) ** 2)
        if error < tolerance:
            break

    return approximated_signal

# Example usage:
# Replace this with your original signal data points
original_signal = np.array([[0.0, 0.0], [0.2, 0.4], [0.4, 0.8], [0.6, 0.6], [0.8, 0.2], [1.0, 0.0]])

d = 3  # Fractional degree of the Bezier curve
tolerance = 1e-6  # Desired tolerance level for stopping the iteration

approximated_signal = nss_approximation(original_signal, d, tolerance)
print("Approximated Signal:")
print(approximated_signal)
