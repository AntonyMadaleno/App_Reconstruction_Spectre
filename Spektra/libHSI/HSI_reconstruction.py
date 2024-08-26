# -*- coding: utf-8 -*-

#====================================================================================================================================================================================================================================================================
# IMPORTS #
#====================================================================================================================================================================================================================================================================

import numpy as np
from interpolation import *
import matplotlib.pyplot as plt
from scipy.optimize import nnls
import numpy as np
from libHSI.HSI_distances import *
from libHSI.HSI_library import *
from tqdm import tqdm
from reconstruction import *

from tkinter import filedialog as fd

#====================================================================================================================================================================================================================================================================

def Derive(F):
    '''
        Return the derivative of a function curve, based on finite derivatives

        ! On suppose les dx normaliser constant

        Input:
        #######

            param   F : The function response
            type    F : 1-D Array
        
        Return:
        #######
            rtype   : 1-D Array
    '''

    D = [ ( F[1] - F[0] ) ]

    # F'(x) ~ ( F(x+1) - F(x) + F(x) - F(x-1) ) / 2
    for x in range(1, F.shape[0] - 1):
        D.append( ( ( F[x+1] - F[x] ) - ( F[x-1] - F[x] ) ) / 2 )
    
    D.append(  F[F.shape[0] - 1] - F[F.shape[0] - 2] )

    return np.array(D)

#====================================================================================================================================================================================================================================================================

def calculate_rmse(y_true, y_pred):
    """
        Calculate the Root Mean Squared Error (RMSE) between two vectors.
        
        Parameters  :
        =============
            :Param y_true   :   The ground truth values.
            :Type  y_true   :   nd-Array

            :Param y_pred   :   The predicted values.
            :Type  y_pred   :   nd-Array
        
        Returns     :
        =============
            :return :   The RMSE value.
            :rtype  :   number
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("The shape of y_true and y_pred must be the same.")
    
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    return rmse

#======================================================================================================================================================================================================================================================

def calculate_score(S1, S2, Definition, Method = "SAM"):
    """
        Calculate a score value based on spectral distance in between two spectras , We assume both spectras are normalized
        
        Parameters  :
        =============
            :Param S1       :   Spectra 1
            :Type  S1       :   1D Array

            :Param S2       :   Spectra 2
            :Type  S2       :   1D Array

            :Param Method   :   The name of the distance used ("SAM", "SGA", "SID", "SCA", "Minkowski_L1", ...)
            :Type  Method   :   String
        Returns     :
        =============
            :return :   The Score Value
            :rtype  :   number
    """

    E1 = np.trapz(S1, dx = Definition)
    E2 = np.trapz(S2, dx = Definition)

    H1 = S1 / E1
    H2 = S2 / E2

    if Method == "SAM":
        return dist_SAM(S1, S2, Definition)
    if Method == "SGA":
        return dist_SGA(H1, H2)
    if Method == "SID":
        return dist_SID(H1, H2)
    if Method == "SCA":
        return dist_SCA(H1, H2)
    if Method == "Minkowski_L1":
        return dist_Minkowski(S1, S2, p = 1, resolution= Definition)
    if Method == "Minkowski_L2":
        return dist_Minkowski(S1, S2, p = 2, resolution= Definition)
    if Method == "Minkowski_Inf":
        return dist_Minkowski(S1, S2, p = int('inf'), resolution= Definition)

#======================================================================================================================================================================================================================================================

# Function to handle the score calculation
def calculate_scores_vectorized(S1, S2_array, Definition, Method):
    """
        Calculate a score value based on spectral distance in between two spectras , We assume both spectras are normalized
        
        Parameters  :
        =============
            :Param S1       :   Spectra 1 (Reference)
            :Type  S1       :   1D Array

            :Param S2_array :   Spectras 2 (the spectras to evaluate)
            :Type  S2_array :   2D Array

            :Param Method   :   The name of the distance used ("SAM", "SGA", "SID", "SCA", "Minkowski_L1", ...)
            :Type  Method   :   String
        Returns     :
        =============
            :return :   The Score Value
            :rtype  :   number
    """

    S1_E    = np.trapz(S1, dx=Definition)
    S2_E    = np.trapz(S2_array, dx=Definition, axis=1)

    H1          = S1 / S1_E
    H2_array    = S2_array / S2_E[:, np.newaxis]

    if Method == "SAM":
        return np.array([dist_SAM(S1, S2_array[i], Definition) for i in range(len(S2_array))])
    if Method == "SGA":
        return np.array([dist_SGA(H1, H2) for H2 in H2_array])
    if Method == "SID":
        return np.array([dist_SID(H1, H2) for H2 in H2_array])
    if Method == "SCA":
        return np.array([dist_SCA(H1, H2) for H2 in H2_array])
    if Method == "Minkowski_L1":
        return np.array([dist_Minkowski(S1, S2_array[i], p=1, resolution=Definition) for i in range(len(S2_array))])
    if Method == "Minkowski_L2":
        return np.array([dist_Minkowski(S1, S2_array[i], p=2, resolution=Definition) for i in range(len(S2_array))])
    if Method == "Minkowski_Inf":
        return np.array([dist_Minkowski(S1, S2_array[i], p=np.inf, resolution=Definition) for i in range(len(S2_array))])

#==================================================================================================================================================================================================================================================================#

def HSI_Rec_NNLS(Ref, Samples, low_norm = 300.0, high_norm = 1000.0, definition = 1.0, debug = False):
    """
        Estimate a coefficient vector K so that K * S = T where S are our spectral responses samples and T the target spectral responses. In this method we aim to this objectif using a non negative least square.

        Parameters :
        ============
            :Param Ref          : The reference spectra
            :Type Ref           : 2D array with Waves (Col 0) and Intensity (Col 1)

            :Param Samples      : The sample spectras used in the estimation
            :Type Samples       : List of 2D arrays with Waves (Col 0) and Intensity (Col 1)

            :Param low_norm     : Lower bound used in the normalization process
            :Type low_norm      : Unsigned Integer

            :Param high_norm    : Higher bound used in the normalization process
            :Type high_norm     : Unsigned Integer

            :Param Definition   : The lowest definition level in 'nm'
            :Type Definition    : Any number > 0

            :Param debug        : Toggle debug (save plots)
            :Type debug         : Boolean

        Return     :
        ============
            :return :   Coefficients obtained trough optimisation
            :rtype  :   1D Array of non negative float
    """

    # Normalisation

    normalized_Ref,  E_target = normalize_wavelength_sampling(Ref, definition, low_norm, high_norm)
    normalized_Ref = normalized_Ref[0]
    E_target = E_target[0]

    normalized_Samples, E_sample = normalize_wavelength_sampling( np.transpose(Samples, (2,0,1)), definition, low_norm, high_norm)

    # Preparation de variables

    scalars = np.zeros(len(normalized_Samples))
    dists   = np.zeros(len(normalized_Samples))
    names   = []
    X       = np.arange(len(normalized_Samples))

    # DEBUG ==================================================================================================================================================================================================== #

    if debug:

        for i in range (0, len(normalized_Samples)):
            scalars[i]  = np.dot(normalized_Ref[1], normalized_Samples[i][1])
            dists[i]    = dist_euclidienne_cum(normalized_Ref[1], normalized_Samples[i][1])
            names.append(normalized_Samples[i][0])

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
        plt.close()

    # DEBUG ==================================================================================================================================================================================================== #

    target  = np.array( normalized_Ref[1] )
    samples = []

    for i in range(len(normalized_Samples)):
        samples.append(np.array(normalized_Samples[i][1]))

    samples = np.array(samples)

    # Solve for K using non-negative least squares

    coefficients, rest = nnls(samples.T, target, maxiter = 10 * definition * (high_norm - low_norm))
    coefficients *= E_target / E_sample

    # DEBUG ==================================================================================================================================================================================================== #

    if debug:

        # Compute estimation results
        res = np.dot(coefficients.T, samples)

        ######## PLOT THE SPECTRAL DISTANCES AS BAR CHART #######
        fig_bar, ax_bar = plt.subplots(figsize=(14,8))

        ax_bar.bar(X, dists, width=0.8, alpha= 0.5, color="#66ff66")
        ax_bar.bar(X[-1] + 1, dist_euclidienne_cum(normalized_Ref[2], res.flatten()), width=0.8, alpha= 0.5, color="#ff0066")
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

        ax_est.plot(normalized_Ref[1], res.flatten(),              lw=3, alpha= 0.5, color="#ff0066")
        ax_est.plot(normalized_Ref[1], normalized_Ref[2], lw=3, alpha= 0.5, color="#00ff66")
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
        plt.close()

    # DEBUG ==================================================================================================================================================================================================== #

    return coefficients, rest

#===================================================================================================================================================================================================================================================================

def HSI_Rec_Succesive(Ref, Samples, low_norm = 300.0, high_norm = 1000.0, definition = 1.0, debug = False):
    """
        Estimate a coefficient vector K so that K * S = T where S are our spectral responses samples and T the target spectral responses. In this method we aim to this objectif using non negative least square succesivly and adding the samples one by one depending
        on their scalar product with the reference, we keep them only if they help decrease the distance between the estimation and the target.

        Parameters :
        ============
            :Param Ref          : The reference spectra
            :Type Ref           : 2D array with Waves (Col 0) and Intensity (Col 1)

            :Param Samples      : The sample spectras used in the estimation
            :Type Samples       : List of 2D arrays with Waves (Col 0) and Intensity (Col 1)

            :Param low_norm     : Lower bound used in the normalization process
            :Type low_norm      : Unsigned Integer

            :Param high_norm    : Higher bound used in the normalization process
            :Type high_norm     : Unsigned Integer

            :Param Definition   : The lowest definition level in 'nm'
            :Type Definition    : Any number > 0

            :Param debug        : Toggle debug (save plots)
            :Type debug         : Boolean

        Return     :
        ============
            :return :   Coefficients obtained trough optimisation
            :rtype  :   1D Array of non negative float
    """

    # Normalisation

    normalized_Ref,    E_target = normalize_wavelength_sampling([Ref], definition, low_norm, high_norm)
    normalized_Ref = normalized_Ref[0]
    E_target = E_target[0]
    normalized_Samples,   E_sample = normalize_wavelength_sampling(Samples,   definition, low_norm, high_norm)

    # Préparation de variables

    scalars = np.zeros(len(normalized_Samples))
    dists_SAM = np.zeros(len(normalized_Samples))
    dists_minkowski = np.zeros(len(normalized_Samples))
    dists_csiszar = np.zeros(len(normalized_Samples))
    names = []

    # Calculate some information about the spectrums compared the reference

    for i in range (0, len(normalized_Samples)):
        scalars[i] = np.dot(normalized_Ref[2], normalized_Samples[i][2])
        names.append(normalized_Samples[i][0])

    # get the decreasing order along the scalars calculated previously

    order = np.flip(np.argsort(scalars))

    target = np.array( normalized_Ref[2] )
    samples = []

    coefficients = []
    rest = []

    x_leds = []
    y_leds = []
    xy_ref = []

    # Calculate chromaticity coordinates (may be used if debug is toggled)

    XYZ_ref = monoSpec_2_CMF_XYZ(normalized_Ref[2], normalized_Ref[1], cmf = 'libHSI\\data\\#_360_830_2deg.npy')
    xy_ref.append( XYZ_ref[0,0] / np.sum(XYZ_ref))
    xy_ref.append( XYZ_ref[0,1] / np.sum(XYZ_ref))
    
    # Add spectrums one by one using the order

    for i in range(len(order)):
        samples.append(np.array(normalized_Samples[order[i]][2]))
        nparr_tmp = np.array(samples)

        # Solve for K using non-negative least squares

        coefficients_tmp, rest_tmp = nnls(nparr_tmp.T, target)
        coefficients.append(coefficients_tmp)
        rest.append(rest_tmp)

        res = np.dot(coefficients_tmp.T, nparr_tmp)

        dists_SAM[i] = dist_SAM(normalized_Ref[2], res.flatten())
        dists_minkowski[i] = dist_Minkowski(normalized_Ref[2], res.flatten())
        dists_csiszar[i] = dist_Csiszar(normalized_Ref[2], res.flatten())

        # DEBUG =======================================================================================================================================================================================

        if debug:
            ######## PLOT THE ESTIMATED SPECTRUM #######
            fig_est, ax_est = plt.subplots(figsize=(14,8))

            ax_est.plot(normalized_Ref[1], res.flatten(),              lw=3, alpha= 0.5, color="#ff0066")
            ax_est.plot(normalized_Ref[1], normalized_Ref[2], lw=3, alpha= 0.5, color="#00ff66")
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

        # DEBUG =======================================================================================================================================================================================

        ###########################################
        
        xy_res = []

        XYZ_led = monoSpec_2_CMF_XYZ(normalized_Samples[order[i]][2], normalized_Samples[order[i]][1], cmf = 'libHSI\\data\\#_360_830_2deg.npy')
        x_leds.append( XYZ_led[0,0] / np.sum(XYZ_led))
        y_leds.append( XYZ_led[0,1] / np.sum(XYZ_led))

        xy_leds = np.array([x_leds, y_leds])

        XYZ_res = monoSpec_2_CMF_XYZ(res, normalized_Ref[1], cmf = 'libHSI\\data\\#_360_830_2deg.npy')
        xy_res.append( XYZ_res[0,0] / np.sum(XYZ_res))
        xy_res.append( XYZ_res[0,1] / np.sum(XYZ_res))

        plot_chromaticity_with_points(
            xy_leds, 
            xy_ref,
            xy_res,
            name = f"chroma_plot_{i}.png"
        )

    X = np.arange(len(order))

    # DEBUG ==================================================================================================================================================================================================== #

    if debug:
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

    # DEBUG ==================================================================================================================================================================================================== #

    arg = np.argmin(rest)
    coefficients[arg] *= E_target / E_sample

    return coefficients[arg], rest
    
#======================================================================================================================================================================================================================================================

def HSI_Rec_Stochastique(Ref, Samples, Criteria = "Minkowski_L2", low_norm = 380, high_norm = 780, definition = 1, steps = 100, nrand = 10000, debug = False):

    """
        Estimate a coefficient vector K so that K * S = T where S are our spectral responses samples and T the target spectral responses. In this method we aim to this objectif using a stochastic approach, trough iterations we test some random values in order
        to optimise a distance (score) given as optimisation criteria

        Parameters :
        ============
            :Param Ref          : The reference spectra
            :Type  Ref          : 2D array with Waves (Col 0) and Intensity (Col 1)

            :Param Samples      : The sample spectras used in the estimation
            :Type  Samples      : List of 2D arrays with Waves (Col 0) and Intensity (Col 1)

            :Param Criteria     : Name of the distance to use as optimisation criteria
            :Type  Criteria     : String

            :Param low_norm     : Lower bound used in the normalization process
            :Type  low_norm     : Unsigned Integer

            :Param high_norm    : Higher bound used in the normalization process
            :Type  high_norm    : Unsigned Integer

            :Param Definition   : The lowest definition level in 'nm'
            :Type  Definition   : Any number > 0

            :Param debug        : Toggle debug (save plots)
            :Type  debug        : Boolean

        Return     :
        ============
            :return :   Coefficients obtained trough optimisation
            :rtype  :   1D Array of non negative float
    """

    # Normalisation
    normalized_target_curve, E_target   = normalize_wavelength_sampling(Ref, low_bound = low_norm, high_bound = high_norm)
    normalized_target_curve = normalized_target_curve[0]
    E_target = E_target[0]

    normalized_sample_curves, E_sample  = normalize_wavelength_sampling( np.transpose(Samples, (2,0,1)), low_bound = low_norm, high_bound = high_norm)

    # Keep energy if minkowsky method is used

    # Non negative least square estimation of the coefficient (our starting point)
    Coeff = np.random.random(len(E_sample))

    samples = []

    for i in range(len(normalized_sample_curves)):
        samples.append(np.array(normalized_sample_curves[i][1]))

    samples = np.array(samples)

    # Compute the score at initialisation (NNLS)
    ans = np.dot(Coeff.T, samples)
    
    score = calculate_score(normalized_target_curve[1], ans, Definition= definition, Method= Criteria)
    Current = ans

    scores_overtime = {
        'Minkowski_L1'  : np.zeros(steps),
        'Minkowski_L2'  : np.zeros(steps),
        'SAM'           : np.zeros(steps),
        'SGA'           : np.zeros(steps),
        'SCA'           : np.zeros(steps),
        'SID'           : np.zeros(steps),
    }

    # Temperature, correspond the intensity of the random mutation applied in the stochastique reasearch process
    T = 2.0

    # Effectue une recherche d'un minimum locale à chaque itération (optimisation du critère distance)
    for step in tqdm(range(0, steps)):

        # Génération de nos vecteurs de mutation (loi aléatoire normale)
        random_array = np.random.random( (nrand, len(Coeff)) ).astype(np.double) * T - ( T/2 )

        # évaluation de nos coefficients mutés en fonction de nôtre critère
        new_coeff   = Coeff
        ans         = np.dot( ( np.abs(random_array + Coeff) ), samples)

        new_scores = calculate_scores_vectorized(normalized_target_curve[1], ans, definition, Criteria)
        k = np.argmin(new_scores)

        # Mise à jour du résultat si le nouveau score est inférieur au précédant
        if ( new_scores[k] < score):
            score       = new_scores[k]
            Current     = ans[k]
            new_coeff   = np.abs( random_array[k,:] + Coeff )
            
        Coeff = new_coeff

        # Memorise l'évolution du score au cours des itérations afin de fournir des outils de debug
        scores_overtime["Minkowski_L1"][step]   = calculate_score(normalized_target_curve[1], Current, Definition= definition, Method= "Minkowski_L1")
        scores_overtime["Minkowski_L2"][step]   = calculate_score(normalized_target_curve[1], Current, Definition= definition, Method= "Minkowski_L2")
        scores_overtime["SAM"][step]            = calculate_score(normalized_target_curve[1], Current, Definition= definition, Method= "SAM")
        scores_overtime["SGA"][step]            = calculate_score(normalized_target_curve[1], Current, Definition= definition, Method= "SGA")
        scores_overtime["SCA"][step]            = calculate_score(normalized_target_curve[1], Current, Definition= definition, Method= "SCA")
        scores_overtime["SID"][step]            = calculate_score(normalized_target_curve[1], Current, Definition= definition, Method= "SID")

        # Mise à jour de la température
        T = T * 0.99

    waves = normalized_target_curve[0]

    K = ( Coeff )

    Current = np.dot(K, samples)

    E_tmp = np.trapz(Current, dx= definition)

    F = E_target / E_tmp
    
    # DEBUG ==================================================================================================================================================================================================== #

    if debug:
        ######## PLOT THE ESTIMATED SPECTRUM #######

        Current = Current * F
    
        D1 = Derive(Current)
        D2 = Derive(D1)

        T1 = Derive(normalized_target_curve[1] * E_target)
        T2 = Derive(T1)

        fig_D0, ax_D0 = plt.subplots(figsize=(14,8))

        ax_D0.plot(waves, Current,  lw=3, alpha= 0.4, color="#ff3300")
        ax_D0.plot(waves, normalized_target_curve[1] * E_target,  lw=3, alpha= 0.4, color="#aa00ff")
        ax_D0.set_ylabel('Y')
        ax_D0.set_xlabel('Walength (nm)')
        ax_D0.set_title('Spectre estimer (Rouge) et Spectre cible (Violet)')

        plt.tight_layout()
        plt.savefig(fd.asksaveasfilename(), format='png', transparent=True)
        plt.close()

        #### 1st degree #### =================================================================================================================================================================================

        fig_D1, ax_D1 = plt.subplots(figsize=(14,8))

        ax_D1.plot(waves, D1,  lw=3, alpha= 0.4, color="#ff3300")
        ax_D1.plot(waves, T1,  lw=3, alpha= 0.4, color="#aa00ff")
        ax_D1.set_ylabel('Y')
        ax_D1.set_xlabel('Walength (nm)')
        ax_D1.set_title('Derive 1 Spectre estimer (Rouge) et Spectre cible (Violet)')

        plt.tight_layout()
        plt.savefig(fd.asksaveasfilename(), format='png', transparent=True)
        plt.close()

        #### 2st degree #### =================================================================================================================================================================================

        fig_D2, ax_D2 = plt.subplots(figsize=(14,8))

        ax_D2.plot(waves, D2,  lw=3, alpha= 0.4, color="#ff3300")
        ax_D2.plot(waves, T2,  lw=3, alpha= 0.4, color="#aa00ff")
        ax_D2.set_ylabel('Y')
        ax_D2.set_xlabel('Walength (nm)')
        ax_D2.set_title('Derivé 2nd Spectre estimer (Rouge) et Spectre cible (Violet)')

        plt.tight_layout()
        plt.savefig(fd.asksaveasfilename(), format='png', transparent=True)
        plt.close()

        return K, scores_overtime

    # DEBUG ==================================================================================================================================================================================================== #

    return K

#======================================================================================================================================================================================================================================================

def HSI_Rec_Recuit_Simuler(Ref, Samples, Criteria = "Minkowski_L2", low_norm = 380, high_norm = 780, definition = 1, steps = 500000, debug = False):

    """
        Estimate a coefficient vector K so that K * S = T where S are our spectral responses samples and T the target spectral responses. In this method we aim to this objectif through succesive stochastic research on different levels of details.

        Parameters :
        ============
            :Param Ref          : The reference spectra
            :Type Ref           : 2D array with Waves (Col 0) and Intensity (Col 1)

            :Param Samples      : The sample spectras used in the estimation
            :Type Samples       : List of 2D arrays with Waves (Col 0) and Intensity (Col 1)

            :Param Criteria     : The distance measure used in the optimization process (See HSI_Distance)
            :Type Criteria      : String

            :Param low_norm     : Lower bound used in the normalization process
            :Type low_norm      : Unsigned Integer

            :Param high_norm    : Higher bound used in the normalization process
            :Type high_norm     : Unsigned Integer

            :Param Definition   : The lowest definition level in 'nm'
            :Type Definition    : Any number > 0

            :Param steps        : The maximum amount of steps performed on each optimisation levels
            :Type steps         : Unsigned int

            :Param nrand        : The amount of random mutations tested on each steps
            :Type nrand         : Unsigned integer

            :Param debug        : Toggle debug (save plots)
            :Type debug         : Boolean

        Return     :
        ============
            :return :   Coefficients obtained trough optimisation
            :rtype  :   1D Array of non negative float
    """

    # Normalisation

    normalized_target_curve, E_target   = normalize_wavelength_sampling(Ref, low_bound = 380, high_bound = 780)
    normalized_target_curve = normalized_target_curve[0]
    E_target = E_target[0]

    normalized_sample_curves, E_sample  = normalize_wavelength_sampling( np.transpose(Samples, (2,0,1)), low_bound = 380, high_bound = 780)

    # Compute the score at initialisation (NNLS)

    RCoeff, __ = HSI_Rec_NNLS(Ref, Samples, low_norm= low_norm, high_norm= high_norm, definition = definition )
    Coeff = np.ones_like(RCoeff)

    samples = []

    for i in range(len(normalized_sample_curves)):
        samples.append(np.array(normalized_sample_curves[i][1]))

    samples = np.array(samples)

    ans = np.dot(Coeff.T, samples)

    # Compute initial score
    score = calculate_score(normalized_target_curve[1], ans, Definition= definition, Method= Criteria)
    Current = ans

    scores_over_time = np.zeros(steps)

    # Keep best coeffs and score in memory
    S = score
    M = Coeff
    T = 2.0

    for step in tqdm(range(0, steps)):
        
        # Génere un vecteur de mutation aléatoire
        random_array = np.random.random( len(Coeff) ).astype(np.double) * T - (T/2)

        new_coeff = Coeff
        ans = np.dot( ( np.abs(random_array + Coeff) ), samples)

        new_score = calculate_score(normalized_target_curve[1], ans, definition, Criteria)

        # Test si le score est inférieur ou un probabilité
        if ( new_score < S or np.random.random() < np.exp(- ( (new_score - score) / T ) ) ):
            score = new_score
            Current = ans
            new_coeff = np.abs( random_array + Coeff )

            if (new_score < S):
                M = new_coeff
                S = new_score
            
        Coeff = new_coeff
        scores_over_time[step] = calculate_score(normalized_target_curve[1], Current, Definition= definition, Method= Criteria)

        if (step % ( steps // 1000 ) ):
            T *= 0.99

    waves = normalized_target_curve[0]

    # DEBUG ==================================================================================================================================================================================================== #

    if debug:
        ######## PLOT THE ESTIMATED SPECTRUM #######
        Ko = M * E_target / np.sum(E_sample)
        Current = np.dot( Ko, samples)

        fig_est, ax_est = plt.subplots(figsize=(14,8))

        ax_est.plot(waves, Current.flatten(),           lw=3, alpha= 0.5, color="#ff0000")
        ax_est.plot(waves, normalized_target_curve[1],  lw=3, alpha= 0.5, color="#ffffff")
        ax_est.set_ylabel('Y')
        ax_est.set_xlabel('Walength (nm)')
        ax_est.set_title('Spectre Cible (Blanc) et Spectre Estimer (Rouge)', color="#f1f1f1")
            
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
        plt.savefig(f"D:/Temp_save/NEW_ALGO.png", format='png', transparent=True)
        plt.close()

        fig_evo, ax_evo = plt.subplots(figsize=(14,8))
        ax_evo.plot( np.arange(steps), scores_over_time, lw = 3, alpha = 0.5, color= "#ff00ff" )

        ### COLOR SETTINGS
        fig_evo.get_axes()[0].spines['bottom'].set_color("#f1f1f1")
        fig_evo.get_axes()[0].spines['top'].set_color("#f1f1f1")
        fig_evo.get_axes()[0].spines['right'].set_color("#f1f1f1")
        fig_evo.get_axes()[0].spines['left'].set_color("#f1f1f1")
        fig_evo.get_axes()[0].xaxis.label.set_color("#f1f1f1")
        fig_evo.get_axes()[0].yaxis.label.set_color("#f1f1f1")
        fig_evo.get_axes()[0].tick_params(axis='x', colors="#f1f1f1")
        fig_evo.get_axes()[0].tick_params(axis='y', colors="#f1f1f1")

        fig_evo.savefig(f"D:/Temp_save/EVO_ALGO.png", transparent= True)
        plt.close()

    # DEBUG ==================================================================================================================================================================================================== #

    return (M * E_target) / E_sample

#====================================================================================================================================================================================================================================================================

def HSI_Rec_Stochastique_Multi_Echelle(Ref, Samples, Criteria = "Minkowski_L2", low_norm = 380, high_norm = 780, definition = 16, levels = 4, steps = 100, nrand = 10000, debug = False):
    
    """
        Estimate a coefficient vector K so that K * S = T where S are our spectral responses samples and T the target spectral responses. In this method we aim to this objectif through succesive stochastic research on different levels of details.

        Parameters :
        ============
            :Param Ref          : The reference spectra
            :Type Ref           : 2D array with Waves (Col 0) and Intensity (Col 1)

            :Param Samples      : The sample spectras used in the estimation
            :Type Samples       : List of 2D arrays with Waves (Col 0) and Intensity (Col 1)

            :Param Criteria     : The distance measure used in the optimization process (See HSI_Distance)
            :Type Criteria      : String

            :Param low_norm     : Lower bound used in the normalization process
            :Type low_norm      : Unsigned Integer

            :Param high_norm    : Higher bound used in the normalization process
            :Type high_norm     : Unsigned Integer

            :Param Definition   : The lowest definition level in 'nm'
            :Type Definition    : Any number > 0

            :Param levels       : The amount of detail levels too optimise
            :Type levels        : Unsigned Char

            :Param steps        : The maximum amount of steps performed on each optimisation levels
            :Type steps         : Unsigned int

            :Param nrand        : The amount of random mutations tested on each steps
            :Type nrand         : Unsigned integer

            :Param debug        : Toggle debug (save plots)
            :Type debug         : Boolean

        Return     :
        ============
            :return :   Coefficients obtained trough optimisation
            :rtype  :   1D Array of non negative float
    """


    # --- Evaluation d'une première estimation par methode d'inversion de matrices (non négative least square)
    RCoeff, __   = HSI_Rec_NNLS( Ref, Samples, low_norm= low_norm, high_norm= high_norm, definition = 1 )
    Coeff = np.ones_like(RCoeff)

    # Normalisation au niveau d'échelle courant
    __, Et  = normalize_wavelength_sampling(Ref, low_bound = 380, high_bound = 780, definition = 1 )
    Et      = Et[0]

    __, Es  = normalize_wavelength_sampling( np.transpose(Samples, (2,0,1)), low_bound = 380, high_bound = 780, definition= 1 )

    # initialise
    A = None
    S = None

    # On effectue une reconstruction pour chaque niveau de resolution
    for level in tqdm( range(0, levels) ):

        R = definition / 2**level

        # Normalisation au niveau d'échelle courant
        normalized_target_curve, E_target   = normalize_wavelength_sampling(Ref, low_bound = 380, high_bound = 780, definition = R )
        normalized_target_curve             = normalized_target_curve[0]
        E_target = E_target[0]

        # --- Samples
        normalized_sample_curves, E_sample  = normalize_wavelength_sampling( np.transpose(Samples, (2,0,1)), low_bound = 380, high_bound = 780, definition= R )

        # Transform Samples to 2D Array
        samples = []

        for i in range(len(normalized_sample_curves)):
            samples.append(np.array(normalized_sample_curves[i][1]))

        samples = np.array(samples)

        # Current answer and score
        if level == 0:
            A   = np.dot(Coeff.T, samples)
            S   = calculate_score(normalized_target_curve[1], A, Definition= R, Method= Criteria)

        else:
            Ax  = np.dot(Coeff.T, samples)
            Sx  = calculate_score(normalized_target_curve[1], Ax, Definition= R, Method= Criteria)

            # Update if new score is lower
            if Sx < S:
                S = Sx
                A = Ax

        # Stochastique research
        T   = 2.00  #   Mutation intensity (lower at each succesive loop)
        C   = True  #   Condition to control the loop
        K   = 0

        while(C):
            # Generate random vectors (mutation)
            random_array = np.random.random( (nrand, len(Coeff)) ).astype(np.double) * T - (T/2)

            # Evaluate current reconstructions
            An = np.dot( ( np.abs(random_array + Coeff) ), samples)

            # Calculate their given scores and extract the index of the best one
            Sn      = calculate_scores_vectorized(normalized_target_curve[1], An, definition, Criteria)
            index   = np.argmin(Sn)

            # Sn - S (évolution du score) 
            G = Sn[index] - S
    
            # Si aucune évolution ou steps_max atteint on arrête la boucle d'optimisation et on passe au niveau d'échelle suivant
            if np.abs(G) < 10**(-9) or K > steps:
                C = False

            # Si une des évolution propose un meilleur score alors on update S (score courant) et A (reconstruction courante)
            elif G < 0:
                A   = An
                S   = Sn[index]

                Coeff = np.abs(Coeff + random_array[index])

            K = K + 1       # Increment K
            T = T * 0.98    # Update T
    
    return (Coeff * Et) / Es