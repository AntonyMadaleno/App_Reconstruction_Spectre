import numpy as np
from interpolation import *
import matplotlib.pyplot as plt
from scipy.optimize import nnls
import numpy as np

from libHSI.HSI_distances import *
from libHSI.HSI_library import *
from colour.plotting import plot_chromaticity_diagram_CIE1931

# Function to plot chromaticity diagram and add points
def plot_chromaticity_with_points(xy_leds, xy_ref, xy_res, name = "chroma_plot.png"):
    ######## PLOT THE ESTIMATED SPECTRUM #######

    # Plot the chromaticity diagram
    fig, ax = plot_chromaticity_diagram_CIE1931(standalone=False)

    ax.plot(np.array(xy_leds)[0,:], np.array(xy_leds)[1,:], 'ro', color="#202020", label="LED_chromaticity") # LEDs
    ax.plot(xy_ref[0], xy_ref[1], 'ro', color="#ff6600", label="Reference_chromaticity") # Ref
    ax.plot(xy_res[0], xy_res[1], 'ro', color="#0066ff", label="Estimation chromaticity") # Res
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    ax.set_title('CIE 1931 Chromaticity Diagram', color="#f1f1f1")
    
    ### COLOR SETTINGS
    fig.get_axes()[0].spines['bottom'].set_color("#f1f1f1")
    fig.get_axes()[0].spines['top'].set_color("#f1f1f1")
    fig.get_axes()[0].spines['right'].set_color("#f1f1f1")
    fig.get_axes()[0].spines['left'].set_color("#f1f1f1")
    fig.get_axes()[0].xaxis.label.set_color("#f1f1f1")
    fig.get_axes()[0].yaxis.label.set_color("#f1f1f1")
    fig.get_axes()[0].tick_params(axis='x', colors="#f1f1f1")
    fig.get_axes()[0].tick_params(axis='y', colors="#f1f1f1")

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"D:/Temp_save/{name}", format='png', transparent=True)

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

    scalars = np.zeros(len(normalized_sample_curves))
    dists = np.zeros(len(normalized_sample_curves))
    names = []
    X = np.arange(len(normalized_sample_curves))

    for i in range (0, len(normalized_sample_curves)):
        scalars[i] = np.dot(normalized_target_curve[2], normalized_sample_curves[i][2])
        dists[i] = dist_euclienne_cum(normalized_target_curve[2], normalized_sample_curves[i][2])
        names.append(normalized_sample_curves[i][0])

    fig_dist, ax_dist = plt.subplots(figsize=(14,8))

    ax_dist.bar(X, scalars, width=0.8, alpha= 0.5, color="#00ff66")
    ax_dist.set_ylabel('Scalaire')
    ax_dist.set_title('Valeur des produit scalaire entre le spectre cible et les spectres d\'estimation', color="#f1f1f1")
    ax_dist.set_xticklabels(names, rotation = 65)

    fig_dist.get_axes()[0].spines['bottom'].set_color("#f1f1f1")
    fig_dist.get_axes()[0].spines['top'].set_color("#f1f1f1")
    fig_dist.get_axes()[0].spines['right'].set_color("#f1f1f1")
    fig_dist.get_axes()[0].spines['left'].set_color("#f1f1f1")
    fig_dist.get_axes()[0].xaxis.label.set_color("#f1f1f1")
    fig_dist.get_axes()[0].yaxis.label.set_color("#f1f1f1")
    fig_dist.get_axes()[0].tick_params(axis='x', colors="#f1f1f1")
    fig_dist.get_axes()[0].tick_params(axis='y', colors="#f1f1f1")

    plt.tight_layout()
    plt.xticks(X - 0.6, names)
    plt.savefig('D:/Temp_save/barchart.png', format='png', transparent=True)

    target = np.array( normalized_target_curve[2] )
    samples = []

    for i in range(len(normalized_sample_curves)):
        samples.append(np.array(normalized_sample_curves[i][2]))

    samples = np.array(samples)

    # Solve for K using non-negative least squares
    coefficients, rest = nnls(samples.T, target, maxiter = 10 * definition * (high_norm - low_norm))

    res = np.dot(coefficients.T, samples)

    ######## PLOT THE SPECTRAL DISTANCES AS BAR CHART #######
    fig_bar, ax_bar = plt.subplots(figsize=(14,8))

    ax_bar.bar(X, dists, width=0.8, alpha= 0.5, color="#66ff66")
    ax_bar.bar(X[-1] + 1, dist_euclienne_cum(normalized_target_curve[2], res.flatten()), width=0.8, alpha= 0.5, color="#ff0066")
    ax_bar.set_ylabel('Scalaire')
    ax_bar.set_title('Distances euclienne_cum entre spectre cible et spectres d\'estimation', color="#f1f1f1")
    ax_bar.set_xticklabels(names, rotation = 65)
    
    ### COLOR SETTINGS
    fig_bar.get_axes()[0].spines['bottom'].set_color("#f1f1f1")
    fig_bar.get_axes()[0].spines['top'].set_color("#f1f1f1")
    fig_bar.get_axes()[0].spines['right'].set_color("#f1f1f1")
    fig_bar.get_axes()[0].spines['left'].set_color("#f1f1f1")
    fig_bar.get_axes()[0].xaxis.label.set_color("#f1f1f1")
    fig_bar.get_axes()[0].yaxis.label.set_color("#f1f1f1")
    fig_bar.get_axes()[0].tick_params(axis='x', colors="#f1f1f1")
    fig_bar.get_axes()[0].tick_params(axis='y', colors="#f1f1f1")

    plt.tight_layout()
    plt.xticks(X - 0.6, names)
    plt.savefig('D:/Temp_save/distchart.png', format='png', transparent=True)

    ######## PLOT THE ESTIMATED SPECTRUM #######
    fig_est, ax_est = plt.subplots(figsize=(14,8))

    ax_est.plot(normalized_target_curve[1], res.flatten(),              lw=3, alpha= 0.5, color="#ff0066")
    ax_est.plot(normalized_target_curve[1], normalized_target_curve[2], lw=3, alpha= 0.5, color="#00ff66")
    ax_est.set_ylabel('Y')
    ax_est.set_xlabel('Walength (nm)')
    ax_est.set_title('Spectre Cible (cyan) et Spectre Estimer (magenta)', color="#f1f1f1")
    
    ### COLOR SETTINGS
    fig_est.get_axes()[0].spines['bottom'].set_color("#f1f1f1")
    fig_est.get_axes()[0].spines['top'].set_color("#f1f1f1")
    fig_est.get_axes()[0].spines['right'].set_color("#f1f1f1")
    fig_est.get_axes()[0].spines['left'].set_color("#f1f1f1")
    fig_est.get_axes()[0].xaxis.label.set_color("#f1f1f1")
    fig_est.get_axes()[0].yaxis.label.set_color("#f1f1f1")
    fig_est.get_axes()[0].tick_params(axis='x', colors="#f1f1f1")
    fig_est.get_axes()[0].tick_params(axis='y', colors="#f1f1f1")

    plt.tight_layout()
    plt.savefig('D:/Temp_save/result.png', format='png', transparent=True)
    plt.show()
    
    return coefficients, rest

def curve_rectruction_scalar(target_curve, sample_curve, low_norm = 300.0, high_norm = 1000.0, definition = 1.0):

    normalized_target_curve = normalize_wavelength_sampling([target_curve], definition, low_norm, high_norm)[0]
    normalized_sample_curves = normalize_wavelength_sampling(sample_curve, definition, low_norm, high_norm)

    scalars = np.zeros(len(normalized_sample_curves))
    dists_SAM = np.zeros(len(normalized_sample_curves))
    dists_minkowski = np.zeros(len(normalized_sample_curves))
    dists_csiszar = np.zeros(len(normalized_sample_curves))
    names = []

    # Calculate some information about the spectrums compared the reference
    for i in range (0, len(normalized_sample_curves)):
        scalars[i] = np.dot(normalized_target_curve[2], normalized_sample_curves[i][2])
        names.append(normalized_sample_curves[i][0])

    # get the decreasing order along the scalars calculated previously
    order = np.flip(np.argsort(scalars))

    target = np.array( normalized_target_curve[2] )
    samples = []

    coefficients = []
    rest = []

    x_leds = []
    y_leds = []
    xy_ref = []

    XYZ_ref = monoSpec_2_CMF_XYZ(normalized_target_curve[2], normalized_target_curve[1], cmf = 'libHSI\\data\\#_360_830_2deg.npy')
    xy_ref.append( XYZ_ref[0,0] / np.sum(XYZ_ref))
    xy_ref.append( XYZ_ref[0,1] / np.sum(XYZ_ref))
    
    # Add spectrums one by one using the order
    for i in range(len(order)):
        samples.append(np.array(normalized_sample_curves[order[i]][2]))
        nparr_tmp = np.array(samples)

        # Solve for K using non-negative least squares
        coefficients_tmp, rest_tmp = nnls(nparr_tmp.T, target)
        coefficients.append(coefficients_tmp)
        rest.append(rest_tmp)

        res = np.dot(coefficients_tmp.T, nparr_tmp)

        dists_SAM[i] = dist_SAM(normalized_target_curve[2], res.flatten())
        dists_minkowski[i] = dist_Minkowski(normalized_target_curve[2], res.flatten())
        dists_csiszar[i] = dist_Csiszar(normalized_target_curve[2], res.flatten())

        ######## PLOT THE ESTIMATED SPECTRUM #######
        fig_est, ax_est = plt.subplots(figsize=(14,8))

        ax_est.plot(normalized_target_curve[1], res.flatten(),              lw=3, alpha= 0.5, color="#ff0066")
        ax_est.plot(normalized_target_curve[1], normalized_target_curve[2], lw=3, alpha= 0.5, color="#00ff66")
        ax_est.set_ylabel('Y')
        ax_est.set_xlabel('Walength (nm)')
        ax_est.set_title('Spectre Cible (cyan) et Spectre Estimer (magenta)', color="#f1f1f1")
        
        ### COLOR SETTINGS
        fig_est.get_axes()[0].spines['bottom'].set_color("#f1f1f1")
        fig_est.get_axes()[0].spines['top'].set_color("#f1f1f1")
        fig_est.get_axes()[0].spines['right'].set_color("#f1f1f1")
        fig_est.get_axes()[0].spines['left'].set_color("#f1f1f1")
        fig_est.get_axes()[0].xaxis.label.set_color("#f1f1f1")
        fig_est.get_axes()[0].yaxis.label.set_color("#f1f1f1")
        fig_est.get_axes()[0].tick_params(axis='x', colors="#f1f1f1")
        fig_est.get_axes()[0].tick_params(axis='y', colors="#f1f1f1")

        plt.tight_layout()
        plt.savefig(f"D:/Temp_save/result_{i}.png", format='png', transparent=True)
        plt.close()

        ###########################################
        
        xy_res = []

        XYZ_led = monoSpec_2_CMF_XYZ(normalized_sample_curves[order[i]][2], normalized_sample_curves[order[i]][1], cmf = 'libHSI\\data\\#_360_830_2deg.npy')
        x_leds.append( XYZ_led[0,0] / np.sum(XYZ_led))
        y_leds.append( XYZ_led[0,1] / np.sum(XYZ_led))

        xy_leds = np.array([x_leds, y_leds])

        XYZ_res = monoSpec_2_CMF_XYZ(res, normalized_target_curve[1], cmf = 'libHSI\\data\\#_360_830_2deg.npy')
        xy_res.append( XYZ_res[0,0] / np.sum(XYZ_res))
        xy_res.append( XYZ_res[0,1] / np.sum(XYZ_res))

        plot_chromaticity_with_points(
            xy_leds, 
            xy_ref,
            xy_res,
            name = f"chroma_plot_{i}.png"
        )

    X = np.arange(len(order))

    ######## PLOT THE SPECTRAL DISTANCES AS BAR CHART #######
    fig_bar, ax_bar = plt.subplots(figsize=(14,8))

    ax_bar.bar(X, dists_SAM, width=0.8, alpha= 0.5, color="#66ff00")
    ax_bar.set_ylabel('SAM score')
    ax_bar.set_xlabel('iteration')
    ax_bar.set_title('Distances SAM entre spectre cible et spectres d\'estimation', color="#f1f1f1")
    
    ### COLOR SETTINGS
    fig_bar.get_axes()[0].spines['bottom'].set_color("#f1f1f1")
    fig_bar.get_axes()[0].spines['top'].set_color("#f1f1f1")
    fig_bar.get_axes()[0].spines['right'].set_color("#f1f1f1")
    fig_bar.get_axes()[0].spines['left'].set_color("#f1f1f1")
    fig_bar.get_axes()[0].xaxis.label.set_color("#f1f1f1")
    fig_bar.get_axes()[0].yaxis.label.set_color("#f1f1f1")
    fig_bar.get_axes()[0].tick_params(axis='x', colors="#f1f1f1")
    fig_bar.get_axes()[0].tick_params(axis='y', colors="#f1f1f1")

    plt.tight_layout()
    plt.savefig("D:/Temp_save/SAM_chart.png", format='png', transparent=True)
    plt.xticks(X - 0.6, X)
    plt.close()

    fig_bar, ax_bar = plt.subplots(figsize=(14,8))

    ax_bar.bar(X, dists_minkowski, width=0.8, alpha= 0.5, color="#0066ff")
    ax_bar.set_ylabel('Minkowski score')
    ax_bar.set_xlabel('iteration')
    ax_bar.set_title('Distances Minkowski entre spectre cible et spectres d\'estimation', color="#f1f1f1")
    
    ### COLOR SETTINGS
    fig_bar.get_axes()[0].spines['bottom'].set_color("#f1f1f1")
    fig_bar.get_axes()[0].spines['top'].set_color("#f1f1f1")
    fig_bar.get_axes()[0].spines['right'].set_color("#f1f1f1")
    fig_bar.get_axes()[0].spines['left'].set_color("#f1f1f1")
    fig_bar.get_axes()[0].xaxis.label.set_color("#f1f1f1")
    fig_bar.get_axes()[0].yaxis.label.set_color("#f1f1f1")
    fig_bar.get_axes()[0].tick_params(axis='x', colors="#f1f1f1")
    fig_bar.get_axes()[0].tick_params(axis='y', colors="#f1f1f1")

    plt.tight_layout()
    plt.savefig("D:/Temp_save/Minkowski_chart.png", format='png', transparent=True)
    plt.xticks(X - 0.6, X)
    plt.close()

    fig_bar, ax_bar = plt.subplots(figsize=(14,8))

    ax_bar.bar(X, dists_csiszar, width=0.8, alpha= 0.5, color="#66ff00")
    ax_bar.set_ylabel('csiszar score')
    ax_bar.set_xlabel('iteration')
    ax_bar.set_title('Distances csiszar entre spectre cible et spectres d\'estimation', color="#f1f1f1")
    
    ### COLOR SETTINGS
    fig_bar.get_axes()[0].spines['bottom'].set_color("#f1f1f1")
    fig_bar.get_axes()[0].spines['top'].set_color("#f1f1f1")
    fig_bar.get_axes()[0].spines['right'].set_color("#f1f1f1")
    fig_bar.get_axes()[0].spines['left'].set_color("#f1f1f1")
    fig_bar.get_axes()[0].xaxis.label.set_color("#f1f1f1")
    fig_bar.get_axes()[0].yaxis.label.set_color("#f1f1f1")
    fig_bar.get_axes()[0].tick_params(axis='x', colors="#f1f1f1")
    fig_bar.get_axes()[0].tick_params(axis='y', colors="#f1f1f1")

    plt.tight_layout()
    plt.savefig("D:/Temp_save/csiszar_chart.png", format='png', transparent=True)
    plt.xticks(X - 0.6, X)
    plt.close()

    ######## PLOT THE SPECTRAL DISTANCES AS BAR CHART #######
    fig_bar, ax_bar = plt.subplots(figsize=(14,8))

    ax_bar.bar(X, rest, width=0.8, alpha= 0.5, color="#00ff66")
    ax_bar.set_ylabel('Reste de l\'inversion')
    ax_bar.set_xlabel('iteration')
    ax_bar.set_title('Reste de l\'inversion en fonction de l\'iteration', color="#f1f1f1")
    
    ### COLOR SETTINGS
    fig_bar.get_axes()[0].spines['bottom'].set_color("#f1f1f1")
    fig_bar.get_axes()[0].spines['top'].set_color("#f1f1f1")
    fig_bar.get_axes()[0].spines['right'].set_color("#f1f1f1")
    fig_bar.get_axes()[0].spines['left'].set_color("#f1f1f1")
    fig_bar.get_axes()[0].xaxis.label.set_color("#f1f1f1")
    fig_bar.get_axes()[0].yaxis.label.set_color("#f1f1f1")
    fig_bar.get_axes()[0].tick_params(axis='x', colors="#f1f1f1")
    fig_bar.get_axes()[0].tick_params(axis='y', colors="#f1f1f1")

    plt.tight_layout()
    plt.savefig("D:/Temp_save/Rest_chart.png", format='png', transparent=True)
    plt.xticks(X - 0.6, X)
    plt.close()

    arg = np.argmin(rest)

    return coefficients[arg], rest