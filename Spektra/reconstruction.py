import numpy as np
from interpolation import *
import matplotlib.pyplot as plt
from scipy.optimize import nnls
import numpy as np
import cv2

from libHSI.HSI_distances import *
from libHSI.HSI_library import *
from colour.plotting import plot_chromaticity_diagram_CIE1931
from tqdm import tqdm

from tkinter import filedialog as fd

#==================================================================================================================================================================================================================================================================#

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

#==================================================================================================================================================================================================================================================================#

def normalize_wavelength_sampling(curves, definition=1.0, low_bound=360.0, high_bound=760.0, interpolation_modality="linear"):
    """
    Normalize wavelength sampling for given curves.

    Parameters:
        curves (list): List of curves, each containing (x_values, y_values).
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
    E = np.zeros(curves.shape[0]).astype(np.double)
    
    for k in range(0, curves.shape[0]):

        y_new = np.zeros(np.size(wl_new))

        X = curves[k,:,0]
        Y = curves[k,:,1]

        x_norm = ( X - np.min(X) ) / (np.max(X) - np.min(X))

        for i in range(0, np.size(wl_new)):

            if (wl_new[i] < np.min(X)):
                y_new[i] = 0.0
            elif(wl_new[i] > np.max(X)):
                y_new[i] = 0.0
            else:
                y_new[i] = linear_interpolation(Y, x_norm, (wl_new[i] - np.min(X)) / (np.max(X) - np.min(X)) )

        Energy = np.trapz(y_new, dx = definition)
        y_new  = y_new / Energy

        E[k] = Energy
        wl_normalized_curves.append( ( np.copy(wl_new), np.copy(y_new) ) )

    return (wl_normalized_curves, E)

#======================================================================================================================================================================================================================================================

def Recalage(Ref, Img):

    # Normalisation
    Ref_norm = ( Ref.astype(float) * 255.0 / np.max(Ref.astype(float)) ).astype(np.uint8)
    Img_norm = ( Img.astype(float) * 255.0 / np.max(Img.astype(float)) ).astype(np.uint8)

    # Utiliser SIFT pour détecter les points d'intérêt et calculer les descripteurs
    sift                        = cv2.SIFT_create()
    keypoints1, descriptors1    = sift.detectAndCompute(Ref_norm, None)
    keypoints2, descriptors2    = sift.detectAndCompute(Img_norm, None)

    # Utiliser le matcher de FLANN pour associer les descripteurs
    index_params    = dict(algorithm=1, trees=5)
    search_params   = dict(checks=50)
    flann           = cv2.FlannBasedMatcher(index_params, search_params)
    matches         = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Appliquer le ratio test de Lowe pour filtrer les bonnes correspondances
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Vérifier qu'il y a assez de bonnes correspondances
    if len(good_matches) > 10:
        # Extraire les points correspondants
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Trouver la matrice homographique
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.USAC_ACCURATE, 5.0)

        print(M)

        # Aligner l'image
        height, width = Ref.shape
        aligned_img = cv2.warpPerspective(Img, M, (width, height))

        return aligned_img
    
    else:
        print("Pas assez de correspondances trouvées - {}/10".format(len(good_matches)))
