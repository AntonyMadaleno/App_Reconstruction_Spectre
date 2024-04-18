import numpy as np
from interpolation import *
import matplotlib.pyplot as plt
from scipy.optimize import nnls
import numpy as np

def normalize_wavelength_sampling(curves, definition=1.0, low_bound=360, high_bound=760, interpolation_modality="linear"):
    """
    Normalize wavelength sampling for given curves.

    Parameters:
        curves (list): List of curves, each containing (name, x_values, y_values, curve_color).
        definition (float): Sampling definition.
        low_bound (int): Lower bound of wavelength.
        high_bound (int): Higher bound of wavelength.
        interpolation_modality (str): Interpolation mode, default is "linear".

    Returns:
        list: List of normalized curves.
    """
    wl_normalized_curves = []

    if low_bound > high_bound:
        tmp = low_bound
        low_bound = high_bound
        high_bound = low_bound

    wl_new = np.linspace(low_bound, high_bound, int((high_bound - low_bound) / definition))
    
    for curve in curves:

        y_new = np.zeros(np.size(wl_new))

        X = curve[1]
        Y = curve[2]

        x_norm = ( X - np.min(X) ) / (np.max(X) - np.min(X))

        for i in range(0, np.size(wl_new)):

            if (wl_new[i] < np.min(X)):
                y_new[i] = 0.0
            elif(wl_new[i] > np.max(X)):
                y_new[i] = 0.0
            else:
                if (interpolation_modality == "linear"):
                    y_new[i] = linear_interpolation(Y, x_norm, (wl_new[i] - np.min(X)) / (np.max(X) - np.min(X)) )

        wl_normalized_curves.append( ( curve[0], np.copy(wl_new), np.copy(y_new), curve[3] ) )

    return wl_normalized_curves


def curve_rectruction(target_curve, sample_curve, low_norm = 300.0, high_norm = 1000.0, definition = 1.0):

    normalized_target_curve = normalize_wavelength_sampling([target_curve], definition, low_norm, high_norm)[0]
    normalized_sample_curves = normalize_wavelength_sampling(sample_curve, definition, low_norm, high_norm)

    target = np.array( normalized_target_curve[2] )
    samples = []

    for i in range(len(normalized_sample_curves)):
        samples.append(np.array(normalized_sample_curves[i][2]))

    samples = np.array(samples)

    # Solve for K using non-negative least squares
    coefficients, __ = nnls(samples.T, target, maxiter = 10 * definition * (high_norm - low_norm))
  
    res = np.dot(coefficients.T, samples)
    plt.plot(normalized_target_curve[1], res.flatten(), label='Estimated Spectrum')
    plt.xlabel('X')
    plt.ylabel('K * L')
    plt.title('Plot of K * L')
    plt.legend()
    plt.show()
    
    return coefficients