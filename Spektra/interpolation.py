from scipy.interpolate import CubicSpline

#interpolation function (Spline)
def spline_interpolation(data, nodes, t):
    """
    Spline Interpolation using Cubic Splines

    Parameters:
    - data: NumPy array of float values.
    - nodes: NumPy array of float values indicating the position of each point in the interval [0.0, 1.0].
    - t: Value between 0 and 1 representing the position where interpolation is needed.

    Returns:
    - Interpolated value
    """
    spline = CubicSpline(nodes, data, axis=2)
    return spline

#interpolation function (Hermite)
def hermite_interpolation(data, nodes, t):
    """
    Hermite Interpolation

    Parameters:
    - data:     NumPy array of float values.
    - nodes:    NumPy array of float values indicating the position of each point in the interval [0.0, 1.0].
    - t:        Value between 0 and 1 representing the position where interpolation is needed.

    Returns:
    - Interpolated value
    """

    n = len(nodes)
    result = 0.0

    for i in range(n):
        # Compute the Hermite basis functions
        basis = 1.0
        for j in range(n):
            if i != j:
                basis *= (t - nodes[j]) / (nodes[i] - nodes[j])

        # Update the result with the contribution of the current data point
        result += data[i] * basis

    return result

#interpolation function (Linear)
def linear_interpolation(data, nodes, t):
    """
    Linear Interpolation

    Parameters:
    - data:     NumPy array of float values.
    - nodes:    NumPy array of float values indicating the position of each point in the interval [0.0, 1.0].
    - t:        Value between 0 and 1 representing the position where interpolation is needed.

    Returns:
    - Interpolated value
    """

    n = len(nodes)
    result = 0.0

    for i in range(n-1):
        if (nodes[i] <= t):
            result = data[i] * abs( ( nodes[i+1] - t ) / ( nodes[i+1] - nodes[i] ) ) + data[i+1] * abs( ( nodes[i] - t ) / ( nodes[i+1] - nodes[i] ) )

    return result
