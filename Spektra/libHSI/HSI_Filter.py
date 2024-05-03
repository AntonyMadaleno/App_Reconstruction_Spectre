# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 00:03:11 2016
    
    Non-Linear Spectral Filtering Functions
    
    Statuts
        - MedianVMF                 : OK (use KLPD as distance) extract the median spectrum from a spectrum array
        - Adapted_MedianVMF         : Ok (Do the same than Median VMF, but the spectral list came from a binary file , adapted for parallelisation)
        - ExtractMedianClassCenter  : Extract the Median from classes of HSI images
        - ParallelExtractMedianClassCenter : Do the same in multiprocessing (OK for python 2.7, not yet with 3.5)
        - BinomialCoeff(n,p):       : Ok, defined the binomial coefficient thanks to the  Pascal's triangle
        - LP_1D_BinomialFilter      : Ok, process a binomial filtering on each spectrum of the image
        - LP_2D_BinomialFilter      : Ok, process a 2D spatial binomial filtering (result = weighted sum of neighobored spectra)
        - LP_3D_BinomialFilter      : Ok, process a 3D filtering : weighting first in the spatial and then in the spectral domain (split the convolution in 2 parts)
        - LP_3D_Conv_BinomialFilter : Ok, process a 3D filtering using a 3D convolution (weighting in the spatial and spectral domain);
                                      take care : the image size is reduced after filtering
       
    Version : 3 (update 1D filtering)    

    Date : 21/12/2018
        
@author: nrichard
"""
import sys
import numpy as np
import scipy as sp
import math  as math # for factorial() (in binomial coeff)


import spectral as spy
import spectral.io.envi as envi       ## for the spectral image load, save and plot

import multiprocessing as multiprocessing

import HSI_distances as distNR
#import HSI_remote_sensing_scene_datas as HSI_data

def MedianVMF(SpectralList , resolution = 1.0):
    '''
    Renvoie le spectre median au sens du VMF (Astola)

    Arguments:
        SpectralList:
            An array of spectra (nb spectral x nb wavelengths)

        Returns:
        SpecLoc : index in the list of the median spectrum
        
        Statut : Ok
    '''
    [NbSpec , NbWaves] = SpectralList.shape
    CumDist = np.zeros( NbSpec, dtype = np.float64)

    for i in np.arange( 0, NbSpec, dtype=np.int32):
        ## copie NbSpec fois le spectre i
        ## calcule sa distace avec les NbSpec
        ## stocke la valeur cumulée
        Ref        = np.tile(SpectralList[i,:], NbSpec).reshape(NbSpec,NbWaves)
        dist2Ref   = distNR.pseudo_div_KL(Ref , SpectralList , resolution)
        CumDist[i] = dist2Ref.sum()
    ## extrait la valeur mini de CumDist
    SpecLoc = np.argmin(CumDist)
    return(SpecLoc)


def AdaptedMedianVMF( Param ):
    '''
    Renvoie le spectre median au sens du VMF (Astola) et le nom du fichier
    Comme les traitements peuvent ne pas être à la même vitesse, cela permet d'associer
    un nom de classe à une coordonnée.

    Arguments:
        Param = 1 liste : [Filename, ClassNb, t_resolution]
        FileName:
            Name of the file containing the array of spectra (nb spectral x nb wavelengths)
            ClassNb : number of the class to produce the return and be able to order them later
            t_resolution : a string chain for the resolution value

        Returns:
            Filename, SpecLoc
            SpecLoc : index in the list of the median spectrum
            
        Statut : Ok mais problème dans l'appel parallélisé.
    '''
    (FileName, ClassNb, t_resolution) = Param
    resolution = np.float(t_resolution)
    SpectralList = np.load(FileName)
    [NbSpec , NbWaves] = SpectralList.shape
    CumDist = np.zeros( NbSpec, dtype = np.float64)

    for i in np.arange( 0, NbSpec, dtype=np.int32):
        ## copie NbSpec fois le spectre i
        ## calcule sa distace avec les NbSpec
        ## stocke la valeur cumulée
        Ref        = np.tile(SpectralList[i,:], NbSpec).reshape(NbSpec,NbWaves)
        dist2Ref   = distNR.pseudo_div_KL(Ref , SpectralList , resolution)
        CumDist[i] = dist2Ref.sum()
    ## extrait la valeur mini de CumDist
    SpecLoc = np.argmin(CumDist)
    return( ClassNb, SpecLoc)

##=============================================================================
def ExtractMedianClassCenter(SpecImg, GT_Img, ImgNb, resolution=1.0, learning_rate=1.0)  :
    '''
        Return an array of 2D-locations providing the median center of each class
        The median are extracted on a subset of the initial class set
        The VMF process is used in order to extract the Median Spectrum

        Be carefull : class 0 = background, don't process it
        Parameters:
        ===========
            :param SpecImg: the Spectral Image
            :type SpecImg: float numpy array of NbLG x NbCol x NbWaves

            :param GT_Img: the GroundTruth Image with the class labels
            :type GT_Img: int numpy array of NbLG x NbCol

            :param ImgNb: Image number in the list of the spectral images (from HSI_remote_sensing_scene_datas.py)            
            :type  ImgNb: int in [0,4]

            :param resolution : spectral sampling in nm (for integration in distance purpose)
            :type resolution: numpy float 

            :param learning_rate : ratio of the used samples from the groundtruth
            :type learning_rate: numpy float in [0, 1.0]
        Returns:
        ========
            :return: MedianClassCenterLocation 
            :rtype: numpy.array (classcount, Lg, Col)
    '''
    
    ClassCoord = np.zeros( (HSI_data.ClassCount[ImgNb],2) , dtype=np.uint16 )
    #For each class: extract the spatial locations in the groundtruth image
    #                keep only a subpart for the median computation (learning) ==> an array of spectra
    #                process the VMF median ==> return the index from the previous table
    # Class 0 = background : don't process it !
    for i in np.arange(1, HSI_data.ClassCount[ImgNb]+1, dtype=np.uint16):
        print("Median Processing : Class "+str(i)+"/"+str(HSI_data.ClassCount[ImgNb]))
        Selected_Pxl  = np.where( GT_Img ==i )  ## extract the locations of class 'i' in the image (be carefull : result= list)

        Selected_Pxl_Count = np.uint16(learning_rate*Selected_Pxl[0].size)
        if (Selected_Pxl_Count < 10):
            Selected_Pxl_Count = 10

        learning_list = np.random.randint(0,Selected_Pxl[0].size, Selected_Pxl_Count) ## select a part of the coordinates ==> a list of coordinates
        Spectrum_list = SpecImg[  Selected_Pxl[0][learning_list], Selected_Pxl[1][learning_list] , 0:HSI_data.MaxWave[ImgNb] ]  ## create the table of spectra

        # extraction des coordonnées du spectre de référence (attention à la liste aléatoire intermédiaire pour l'apprentissage)
        RefIndex      = MedianVMF(Spectrum_list, resolution) ## return the index of the median in the list
        ClassCoord[i-1,:] = np.array( (Selected_Pxl[0][learning_list[RefIndex]], Selected_Pxl[1][learning_list[RefIndex]])) ## coord dans l'image
    
    return ClassCoord
##=============================================================================

##=============================================================================
def ParallelExtractMedianClassCenter(SpecImg, GT_Img, ImgNb, resolution=1.0, learning_rate=1.0)  :
    '''
        Same function than the previous one, but searching the median of each class 
        in parallel processes managed by a polling system.
        
        Return an array of 2D-locations providing the median center of each class
        The median are extracted on a subset of the initial class set
        The VMF process is used in order to extract the Median Spectrum

        Parameters:
        ===========
            :param SpecImg: the Spectral Image
            :type SpecImg: float numpy array of NbLG x NbCol x NbWaves

            :param GT_Img: the GroundTruth Image with the class labels
            :type GT_Img: int numpy array of NbLG x NbCol

            :param ImgNb: Image number in the list of the spectral images (from HSI_remote_sensing_scene_datas.py)            
            :type  ImgNb: int in [0,4]

            :param resolution : spectral sampling in nm (for integration in distance purpose)
            :type resolution: numpy float 

            :param learning_rate : ratio of the used samples from the groundtruth
            :type learning_rate: numpy float in [0, 1.0]
        Returns:
        ========
            :return: MedianClassCenterLocation 
            :rtype: numpy.array (classcount, Lg, Col)
    '''
    nbCPU= 6    ## Adapted for Noel processor
    
    ClassCoord = np.zeros( (HSI_data.ClassCount[ImgNb],2) , dtype=np.uint16 )
    #For each class: extract the spatial locations in the groundtruth image
    #                keep only a subpart for the median computation (learning) ==> an array of spectra
    #                save the spectral list in a numpy file for parallel processing (following step)
    #
    # Don't process the Class '0' : class of unlabelled pixels (without ground-truth)
    for i in np.arange(1, HSI_data.ClassCount[ImgNb]+1, dtype=np.uint16):
       
        # extract randomly "learning_rate % of the spectra owning to the greoundtruth of the class i:
        Selected_Pxl = np.where( GT_Img == i ) ## extract the locations of class 'i' in the image (be carefull : result= list)

        Selected_Pxl_Count = np.uint16(learning_rate*Selected_Pxl[0].size)
        if (Selected_Pxl_Count < 10):
            Selected_Pxl_Count = 10

        learning_list = np.random.randint(0,Selected_Pxl[0].size, Selected_Pxl_Count) ## select a part of the coordinates ==> a list of coordinates
        Spectrum_list = SpecImg[  Selected_Pxl[0][learning_list], Selected_Pxl[1][learning_list] , 0:HSI_data.MaxWave[ImgNb] ]  ## create the table of spectra
    
        print("Median Processing : Class "+str(i)+"/"+str(HSI_data.ClassCount[ImgNb]) + ', used pixels = ', learning_list.size, '/ ', Selected_Pxl[0].size)

        # save the spectral list in a binary file
        filename = "temp/Spectral_list_classe_"+str(i)
        np.save( filename , Spectrum_list)   
        np.save("temp/Selected_Pxl_"+str(i), Selected_Pxl)
        np.save("temp/learning_list_"+str(i), learning_list)
 
    ## create the sequence of arguments to call by the function
    data = [ ["temp/Spectral_list_classe_" + str(i)  +".npy", i, resolution] for i in range(1,HSI_data.ClassCount[ImgNb]+1)]
    
    ## Create the pool and obtain the return as a list of the return (in the order of process ending)
    ## then pool the process of the VMF median ==> return the index from the previous table

    p = multiprocessing.Pool(nbCPU)                 # Spécifie le nombre de CPU
    result = p.map(AdaptedMedianVMF  ,  data )   # génère et gère la queue d'appel 

    np.save("temp/result", result)
    print(result)

    # extraction des coordonnées du spectre de référence (attention à la liste aléatoire intermédiaire pour l'apprentissage)
    ## Extract in the random list the image coordinate of the median spectrum using his list index
    for i in np.arange(0, HSI_data.ClassCount[ImgNb], dtype=np.uint16):
        # classe = result[i][0], sample = result[i][1]
        Selected_Pxl = np.load("temp/Selected_Pxl_"+str(result[i][0])+".npy")
        learning_list = np.load("temp/learning_list_"+str(result[i][0])+".npy")
        ClassCoord[i,:] = np.array( (Selected_Pxl[0][ learning_list[result[i][1]] ], Selected_Pxl[1][learning_list[result[i][1]]]  )) ## coord dans l'image

    return ClassCoord
##=============================================================================

##=============================================================================
def ExtractClosestSpectra( SpecImg, MedianSpectra, NbExtractedSpectra, GroundTruthImg, resolution = 1.0):
    '''
    Extract the Nb closest spectra from SpecImg. 
    If a groundtruth image is defined the extracted spectra comes from each median class

    Arguments:
        SpecImg:
            A spectral Image

        MedianSpectra:
            Table of spectral Median, one line per class

       NbExtractedSpectra: 
           Number of spectra to keep
           
       GroundTruthImg:
           image of groundtruth (classe from 1 to n), take care in Pavia case
           
        Returns:
        Spec : the table of extracted spectra
               np.array( (NbClasses, Nb Spectra to extrat/classe, Nb wavelength))
    '''    
##=============================================================================
    (NbClasses, NbWaves) = MedianSpectra.shape
    SpectralList       = np.zeros ( (NbClasses, NbExtractedSpectra, NbWaves) )
    
    for i in range(NbClasses):
        # extract the spectra of the Class(i)
        Selected_Pxl  = np.where( GroundTruthImg == i+1 )  ## extract the locations of class 'i' in the image (be carefull : result= list)
        Selected_Spectra = SpecImg[ Selected_Pxl[:][0], Selected_Pxl[:][1], : ]
        
        #duplication of the reference Median spectrum and distance processing
        MedianCopies = np.tile( MedianSpectra[i], len(Selected_Pxl[0])).reshape( len(Selected_Pxl[0]), NbWaves )
        SpectralDist = distNR.pseudo_div_KL2(MedianCopies, Selected_Spectra, resolution)
        ClosestIndex = SpectralDist.sum(axis=1).argsort() #return the index relative 
        
        ## A gérer : le cas ou NbExtractedSpectra < Nb Spectra of class(i)
        SpectralList[i, 0:NbExtractedSpectra, :] = Selected_Spectra[ClosestIndex[0:NbExtractedSpectra] , : ]
     
    return (SpectralList[0:NbClasses, 0:NbExtractedSpectra, 0:NbWaves])
##=============================================================================
##=============================================================================
def BinomialCoeff(n,p):
    '''
        Process a binomial coefficient C(n,p) from the factorial expression
        
        ( n )            n!
        (   ) = --------------------
        ( p )      (n-p)! . p !
        
        Arguments:
            n : integer value
            p : integer value lowest to p
            
        Returns :
            the binomial coefficient
    '''
    if (p>n) : 
        sys.exit("BinomialCoeff: n must be > P !")  ## The factotial function will send an error msg
    
    return math.factorial(n) / ( math.factorial(n-p) * math.factorial(p))
##=============================================================================

##=============================================================================
def LP_1D_BinomialFilter(SpecImg, size=3):
    '''
    Process a Low-Pass Filter on the spectra from the HSI passed as arguments.
    The used filter is a binomial one, approximating a Gaussian filter using 
    integers.
    
    Arguments:
        SpecImg: The spectral image to process (Array  of nbLg x nbCol x nbWaves of np.float32)
        Size   : Filter size (2n+1), more larger is the filter, more efficient is the filtering but more costly they are.

    Returns:
        FilSpecImg : The Filtered spectral image (Array  of nbLg x nbCol x nbWaves of np.float32)
        
    Comment: Don't respect the wavelengths correlations 
        
    Statut : Ok

    '''
    if (np.uint(size)/2 == 0):
        print('The filter size must be in 2xn+1')
        return(-1)
    ##------------------------------------------------------------------------- 
    ## Initial Step : construct the Binomial Coeeficient by dot product of 1D coefficients
    ##                                      Process the Binomial 1D coefficient 
    BinomCoeff1D = np.zeros( size, dtype=np.float32)
    for i in range(size):
        BinomCoeff1D[i] = float(BinomialCoeff(size-1, i))

    ##------------------------------------------------------------------------- 
    ## Apply the filter adding the wieghted values of neighborhooded radiance/reflectances
    ## then normalise by the sum of binomial coefficients.

    [NbLg, NbCol, NbWaves] = SpecImg.shape
    FiltSpecImg = np.zeros( (NbLg, NbCol, NbWaves+size-1), dtype = np.float32)
    WeightSum_1D   = np.zeros( (NbLg, NbCol, NbWaves+size-1), dtype = np.float32)
    
    for i in range(size):
            FiltSpecImg[ : , :, i:i+NbWaves ] +=  np.multiply(BinomCoeff1D[i], SpecImg, dtype= np.float32)
            WeightSum_1D[ : , :, i:i+NbWaves ] += BinomCoeff1D[i] 

    FiltSpecImg = np.divide(FiltSpecImg,WeightSum_1D, dtype= np.float32)
    
    BorderSize = np.uint16((size-1)/2)    ## integer division 

    return(FiltSpecImg[ : , : , BorderSize:NbWaves+BorderSize ])
##=============================================================================

##=============================================================================
def LP_2D_BinomialFilter(SpecImg , size=3):
    '''
    Process a Low-Pass Filter in the spatial domain considering a Binomial Filter of size 'size'.
    The result is for each pixel an average spectrum obtained by weighted sum of the
    neighborhooded spectra. 
    (Binomial filters approximate Gaussian filters, larger are the spatial width,
    sharper are the frequencial cut-off)

    Arguments:
        SpecImg: The spectral image to process (Array  of nbLg x nbCol x nbWaves of np.float32)
        Size   : Filter size (2n+1), more larger is the filter, more efficient is the filtering but more costly they are.

        Returns:
        FilSpecImg : The Filtered spectral image (Array  of nbLg x nbCol x nbWaves of np.float32)
        
        Comment: Don't respect the wavelengths correlations 
        
        Statut : Ok
    '''
    ##                                 |  1    4    6    4    1 |
    ##      |  1   2  1 |              |  4   16   24   16   4  |  
    ## 1/16 |  2   4  2 |       1/256  |  6   24   36   24   6  |    
    ##      |  1   2  1 |              |  4   16   24   16   4  |  
    ##                                 |  1    4    6    4    1 |
    
#    Mat = np.array( [[ 1.0,  2.0,   1.0],
#                     [ 2.0,  4.0,   2.0],
#                     [ 1.0,  2.0,   1.0] ])
#    Weight = 16.0
#   
    if (np.uint(size)/2 == 0):
        print('The filter size must be in 2xn+1')
        return(-1)
    ##------------------------------------------------------------------------- 
    ## Initial Step : construct the Binomial Coeeficient by dot product of 1D coefficients
    ##                                      Process the Binomial 1D coefficient 
    BinomCoeff1D = np.zeros( (1,size), dtype=np.float32)
    for i in range(size):
        BinomCoeff1D[0,i] = float(BinomialCoeff(size-1, i))
    ##                            Create the 2D matrix of Binomial coefficients
    Mat = np.dot(BinomCoeff1D.T, BinomCoeff1D)
    
    ##------------------------------------------------------------------------- 
    ## Apply the filter by the sum of the weighted sliced images followed by the normalisation
    [NbLg, NbCol, NbWaves] = SpecImg.shape
    FiltSpecImg = np.zeros( (NbLg + size-1, NbCol + size-1, NbWaves) , dtype = np.float32)
    WeightSum_2D   = np.zeros( (NbLg + size-1, NbCol + size-1, NbWaves) , dtype = np.float32)
    
    for i in range(Mat.shape[0]):
        for j in range(Mat.shape[1]):
            FiltSpecImg[ i:i+NbLg , j:j+NbCol, : ] += Mat[i,j] * SpecImg 
            WeightSum_2D[   i:i+NbLg , j:j+NbCol, : ] += Mat[i,j] 
#    FiltSpecImg /= Mat.sum()
    FiltSpecImg = np.divide(FiltSpecImg,WeightSum_2D)
    
    BorderSize = np.uint16((size-1)/2)    ## integer division 
    return(FiltSpecImg[ BorderSize:NbLg+BorderSize, BorderSize:NbCol+BorderSize, :  ])
##=============================================================================

##=============================================================================
def LP_3D_Conv_BinomialFilter(SpecImg , size=3):
    '''
    True 3D Convolution : used a 3D weighting matrix. 
    Induced problem : this current version require a triple loop !
    for image 100 x 1600 x 848 lambda = processing time = 848 s
    rather than 32 s using the 3D version (spatial 2D followed by a 1D spectral)
    
    Process a Low-Pass Filter considering a Binomial Filter of size 'size'. The filter 
    is processed for each pixel location and channel value in order to preserve
    the inter-wavelength correlation.
    
    (Binomial filters approximate Gaussian filters, larger are the spatial width,
    sharper are the frequencial cut-off)
    
    Take CARE : the final image is smallest than the initial one (filtering limits)

    Arguments:
        SpecImg: The spectral image to process (Array  of nbLg x nbCol x nbWaves of np.float32)
        Size   : Filter size (2n+1), more larger is the filter, more efficient is the filtering but more costly they are.

        Returns:
        FilSpecImg : The Filtered spectral image (Array  of nbLg x nbCol x nbWaves of np.float32)
        
        Comment: The filter is a cube applied at each (x,y) location and for each lambda,
                 The computational cost is important: Parallel processig would be developed 
                 in a second step.
        
        Statut : In Course
    '''
    if (np.uint(size)/2 == 0):
        print('The filter size must be in 2xn+1')
        return(-1)

    ##------------------------------------------------------------------------- 
    ## Initial Step : construct the Binomial Coeeficients by dot product of 1D coefficients
    ##                                      Process the Binomial 1D coefficient 
    BinomCoeff1D = np.zeros( (1,size), dtype=np.uint16)
    Mat          = np.zeros( (size, size, size), dtype = np.float32)
    for i in range(size):
        BinomCoeff1D[0,i] = BinomialCoeff(size-1, i)
    
    ##   Create the 3D matrix of Binomial coefficients : Ok
    Mat2D = np.dot(BinomCoeff1D.T, BinomCoeff1D)
    Mat[:,:,0] = Mat2D
    for i in range(size):
        for j in range (size):
            Mat[i,j,:] = Mat[i,j,0] * BinomCoeff1D

    Normalization_factor = Mat.sum()
    ##------------------------------------------------------------------------- 
    ## Apply the filter : Convolve the filter in the 3 dimension : x, y, lambda
    ## Take care the filtered image is smallest than the initial one !
    ##-------------------------------------------------------------------------
    [NbLg, NbCol, NbWaves] = SpecImg.shape
    FiltSpecImg = np.zeros( (NbLg + size, NbCol + size, NbWaves) )
    BorderSize   = np.uint16((size-1)/2) ## to identify the support of the image to filter
    
    for i in np.arange(BorderSize, NbLg - BorderSize ):  ##+ 2*BorderSize):
        for j in np.arange(BorderSize, NbCol - BorderSize ):  ##+ 2*BorderSize):
            for k in np.arange(BorderSize, NbWaves - BorderSize ):  ##+ 2*BorderSize):
                ## multiplication terme à terme de la matrice 3D avec le cube centrée en i,j,k
                FiltSpecImg[ i, j, k ] = np.sum(Mat * SpecImg[ i-BorderSize : i+ BorderSize+1, 
                                                               j-BorderSize : j+ BorderSize+1, 
                                                               k-BorderSize : k+ BorderSize+1])/Normalization_factor
    ## Limits in the following line modified by NR, the 3/6/2018
    return(FiltSpecImg[ BorderSize:(NbLg-BorderSize), BorderSize:(NbCol-BorderSize),   BorderSize:(NbWaves-BorderSize)])

##=============================================================================
def LP_3D_BinomialFilter(SpecImg , size=3):
    '''
    Process a Low-Pass Filter considering a Binomial Filter of size 'size'. The filter 
    is processed for each pixel location and channel value in order to preserve
    the inter-wavelength correlation.
    

    Arguments:
        SpecImg: The spectral image to process (Array  of nbLg x nbCol x nbWaves of np.float32)
        Size   : Filter size (2n+1), more larger is the filter, more efficient is the filtering but more costly they are.

        Returns:
        FilSpecImg : The Filtered spectral image (Array  of nbLg x nbCol x nbWaves of np.float32)
        
        Comment: The filter is a cube applied at each (x,y) location and for each lambda,
                 The computational cost is important: Parallel processig would be developed 
                 in a second step.
        
        Statut : In Course
    '''
    if (np.uint(size)/2 == 0):
        print('The filter size must be in 2xn+1')
        return(-1)
    
    ## Convolve in the spatial domain first
    FiltImg = LP_2D_BinomialFilter(SpecImg , size)
    ## Convolve in the spectral domain in a second step
    FiltImg = LP_1D_BinomialFilter(FiltImg , size)
    
    return(FiltImg)