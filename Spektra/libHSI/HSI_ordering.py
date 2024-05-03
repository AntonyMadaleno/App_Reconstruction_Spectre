# -*- coding: utf-8 -*-
"""
Created on Wed Mar 07 10:18:02 2018

    HSI_ordering.py
    
    Purpose : Ordering functions of spectrum list

    Functions :
        open_mat_Spec_img : open a matlab file and extract the spectral data
        open_envi_image   : open an envy file using the hdr file
        save_ENVI_Img     : save an array as en envy file (including the hdr file)

        Process_Reflectance: process the reflectance from the white and black spectral images

        Spec2ENVI_RGB     : return a RGB image in Envy mode
        Spec2CMF_XYZ      : return a XYZ image using the CIE CMF XYZ
        XYZ_2_RGB         : return a RGB image using existing RGB primaries (adobe, sRGB, ...)
        Spec_2CMF_RGB     : return a RGB image using the CIE CMF
        Spec2RGB_MF       : return a RGB image using a particular CMF

        plot_3Dellispse
        plot_2DEllipse
        plot_BHSD         
    
    status : Ok

    date   : 07/03/2018
    
@author: nrichard
"""
import numpy                             as np

import spectral                          as spy
import spectral.io.envi                  as envi       ## for the spectral image load, save and plot

## HSI Libraries
import HSI_distances                     as HSI_dist

################################################################################
################################################################################
################################################################################

##=============================================================================
## Fonctions locales
##=============================================================================
##=============================================================================
def CRA_Ordering(SpectralList , O_Plus_Infty, O_Minus_Infty, resolution = 1.0):
    '''
    return the rank of the spectra inside the providen spectral list to extract
    the spectral maximum, minimum, median or rank
    
    Arguments:
        SpectralList:
            An array of spectra (nb spectral x nb wavelengths)
        O_Plus_infty:
            A spectrum (array of float)
        O_Moins_infty:
            A spectrum (array of float)
        Resolution :
            a float (spectral resolution for the spectral distance processsing)

        the median can be obtained using np.median
        
        Returns:
        CRA_rank : rank of the spectra from the closest to O^-\infty to O^+\infty
                    CRA_rank[0]               : give the index of the min
                    CRA_rank[CRA_rank.size-1] : give the index of the max
                    CRA_rank[ CRA_rank.size // 2 ] : give the index of the median
                    
        Statut : in course
    '''
    [NbSpec , NbWaves] = SpectralList.shape
    
    ## process dist2PlusInfty = d(Si, O^+\infty)
    O_List = np.tile(O_Plus_Infty, NbSpec).reshape(NbSpec, NbWaves)
    dist2PlusInfty = HSI_dist.pseudo_div_KL2(SpectralList, O_List, resolution).sum(axis=1)
    
    ## process dist2MinusInfty = d (Si, O^-\infty)
    O_List = np.tile(O_Minus_Infty, NbSpec).reshape(NbSpec, NbWaves)
    dist2MinusInfty = HSI_dist.pseudo_div_KL2(SpectralList, O_List, resolution).sum(axis=1)
    
    ## process dOO = d(O^+\infty, O^-\infty)
    dOO = HSI_dist.pseudo_div_KL2(O_Minus_Infty , O_Plus_Infty , resolution).sum(axis=1)
    
    ## process R0 = dist2PlusInfty / dist2MoinsInfty
    ## R0 = dist2PlusInfty / dist2MinusInfty
    
    ## process R1 = 1.0 / R0
    ## R1 = 1.0 / R0
    R1 =  dist2MinusInfty / dist2PlusInfty 
      
    ## process R2 = 2.0 * dist2MinusInfty  / dOO
    R2 = 2.0 * dist2MinusInfty / dOO
    
    ## CRA_rank = np.lexsort( (R2, R1) ) (most important at the right)
    ## if R0(S1) > R0(S2) then R1(S1) < R1(S2)
    CRA_rank = np.lexsort( (R2, R1) )
    
    return CRA_rank        
##=============================================================================
