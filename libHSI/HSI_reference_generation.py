# -*- coding: utf-8 -*-
"""
Created on Tue Mar 07 14:30:02 2017
    HSI_reference_generation.py
    
    Purpose : functions generating set of spectra using simple synthetic model

    Functions :
        Spectrum_generation : create a set of spectra from gaussian fct ('exp', 'erf', 'erfc', 'gauss')
        MinMaxReference     : extract the marginal Min or Max for a set of spectra
        
        
        recent add : 
            - Gaussian function (magnitude is relative to the standard-deviation : preserve the total energy)
            - Noise model for each lambda based on a Weibull distribution
    
    status : OK

    date   : 16/06/2018
    
    Remarks : update due to modification of scipy 1.1.0 (7/06/2016)
              Min(), Max marginal reference extraction (16/06/2018)
    
@author: nrichard
"""
import numpy as np
import scipy as sp
from   scipy import special

################################################################################
################################################################################
################################################################################

##=============================================================================
def Spectrum_generation(count, wavelengths, data = 450.0, Sigma =15.0, Magnitude =0.7, Offset=0.0,  
                                           Var_Mu =0.0, Var_Sigma =0.0, Var_Magn =0.0, Var_Offset=0.0, 
                                           Noise_Mean = 0.0, Noise_Var =0.0,
                                           type_fct='exp'):
    '''
    create a set of 'count' spectra with the fixed parameter and a variable part from normal random distribution.
    type fct= 'exp', 'erf', 'erfc', 'gauss'
    
    A random noise (Weibull Distribution) is added to obtain a more realistic behaviour for statistical analysis

    Arguments:
        count:
            The number of spectra to generate

        wavelengths:
            the table of wavelengths for the wavelength range and resolution

        data:
            the center or maximal value location of the gaussian/sigma function
            or
            an array with the expected values for the generation

        Sigma :
            the standard-deviation

        Magnitude:
            The magnitude of the gaussian function

        Offset :
            The intensity Offset

        Var_Mu, Var_Sigma, Var_Magn, Var_Offset:
            The magnitude of the random normal variation around the initial value
            
        Noise_Mean, Noise_Var : 
            parameters for the noise model at each lambda (Weibull distribution)

        type_fct:
            nature of the generated function : gaussian_like, erf (integral of Gaussian),
            erfc (1- integral of Gaussian function), gaussian
        Returns:
        Spec : the table of spectra
    '''
    Spec       = np.zeros( (count,wavelengths.size), dtype=np.double )     ## the spectra table to retur,
    list_fct= ['exp', 'erf', 'erfc', 'gauss']
    
    ## Adaptation to the parameter type
    if (type(data) == list ):
        # optimized case : parameters are given from an array
        Mu_Lambda = data[0]; Sigma     = data[1]; Magnitude = data[2]; Offset=data[3];  
        Var_Mu    = data[4]; Var_Sigma = data[5]; Var_Magn  = data[6]; Var_Offset=data[7];
        Noise_Mean= data[8]; Noise_Var = data[9];
        type_fct  = data[10]

    elif (np.isscalar(data)== False):
        # optimized case : parameters are given from an array
        Mu_Lambda = data[0]; Sigma     = data[1]; Magnitude = data[2]; Offset=data[3];  
        Var_Mu    = data[4]; Var_Sigma = data[5]; Var_Magn  = data[6]; Var_Offset=data[7];
        Noise_Mean= data[8]; Noise_Var = data[9];
        type_fct  = list_fct[np.uint16(data[10])]
    else:
        # standard case : all the parameters are providen
        Mu_Lambda = data
    
    ## construction of the spectral set
    Rand_Mu    = Var_Mu * np.random.randn( count,1 )
    Rand_Sigma = Var_Sigma *np.random.randn( count,1 )
    Rand_Magn  = Var_Magn *np.random.randn( count,1 )
    Rand_Offset= Var_Offset *np.random.randn( count,1 )
    

    if (type_fct == 'exp'):
        Spec = Rand_Offset + Offset + (Magnitude+Rand_Magn)*np.exp(-(wavelengths - Mu_Lambda-Rand_Mu)*(wavelengths - Mu_Lambda-Rand_Mu) / ( (Sigma+Rand_Sigma) * (Sigma+Rand_Sigma) ) )
    elif (type_fct == 'erf'):
        Spec = Rand_Offset + Offset + (Magnitude+Rand_Magn)*(0.5+0.5*special.erf( (wavelengths - Mu_Lambda-Rand_Mu) / (Sigma+Rand_Sigma) ) )
    elif (type_fct == 'erfc'):
        Spec = Rand_Offset + Offset + (Magnitude+Rand_Magn)*(0.5*special.erfc( (wavelengths - Mu_Lambda-Rand_Mu) / (Sigma+Rand_Sigma) ) )
    elif (type_fct == 'gauss'):
        Spec = Rand_Offset + Offset + (Magnitude+Rand_Magn)* np.exp(-(wavelengths - Mu_Lambda-Rand_Mu)*(wavelengths - Mu_Lambda-Rand_Mu) / (2* (Sigma+Rand_Sigma) * (Sigma+Rand_Sigma) ) ) /((Sigma+Rand_Sigma)*np.sqrt(2*np.pi))

    if (Noise_Var > 0):
        Spec = Spec + Noise_Mean * np.random.weibull(Noise_Var, Spec.shape  )
    return (Spec)
##=============================================================================


##=============================================================================
def MinMaxReference(SpecImg, type_fct = 'min'):
    '''
        Extract the marginal minimum (or maximum) spectrum from a spectrum set.
        Marginal means that an extremum (min or max) is extracted from each 
        spectral channel/wavelength.
        
        Arguments:
            SpecImg : the 2D or 1D array of spectra
            type_fct : 'min' or 'max' or 'minmax' string.
            
        Returns:
            Extremum : Minimum or maximum spectrum or [min, max] in the case of 'minmax'
    '''
    if (SpecImg.ndim == 3): ## Image case
        if (type_fct =='min'):
            return SpecImg.min(axis=0).min(axis=0)
        elif (type_fct =='max'):
            return SpecImg.max(axis=0).max(axis=0)
        else:
            return [SpecImg.min(axis=0).min(axis=0) , SpecImg.max(axis=0).max(axis=0)]
        
    elif (SpecImg.ndim == 2):
        if (type_fct =='min'):
            return SpecImg.min(axis=0)
        elif (type_fct =='max'):
            return SpecImg.max(axis=0)
        else:
            return [SpecImg.min(axis=0) , SpecImg.max(axis=0)]
    else:
        print('Error MinMaxReference : Argument SpecImg not an array of spectra')
        return(-1)
##=============================================================================

        