# -*- coding: utf-8 -*-
"""
Created on Wed May 30 11:55:02 2018

    Name   : HSI_texture.py
    Date   : 16/04/2021
    Status : in Course/Validation
    
    Purpose : Texture Features for HSI

    Functions :
    -----------
        SDOM                    : Create the Spectral Difference Occurrence Matrix(SDOM)
        Ellipse_2D              : Identify a 2D distribution as an ellipsoid
        Hurst_Processing        : Process the Hurst coefficient from an image
        Tab_Hurst_Processing    : Split the image in sub-image before Hurst processing
        
        

    Remarks : 
    ---------
        Hurst processing added in April 2021 : to validate
    
        the proposed version of the Spectral difference occurrence matrix
        produces a feature defined in the positive quarter of plane 
        (shape and intensity differences are positive). 
        It is possible to modify the definition in order to use a full plane
        (embedding a spectral reference to have a difference indicating if
         the measure is lowest ==> negative or highest (==> positive) thanks to an external reference)
        
        To change / adapt the resolution variable and to replace by the wavelength list
        for non-regular spectral sampling.
        
        Require the global variables : Control_plot and Control_print for debug / supervision



@author: nrichard
"""
import numpy                                  as np
from scipy                 import stats
from matplotlib            import pyplot      as plt
from matplotlib.patches    import Ellipse

import spectral                               as spy
import spectral.io.envi                       as envi       ## for the spectral image load, save and plot

## HSI Libraries
import HSI_distances                          as HSI_dist


################################################################################
################################################################################
################################################################################


################################################################################
################################################################################
#####                                                                       ####
#####           SDOM : Spectral Difference Occurence Matrix                 ####
#####                                                                       ####
################################################################################
################################################################################

def SDOM(SpecImg, Spatial_dist = 1, Direction = 0, Resolution =1.0):
    '''
    Return the 2-dimensional array of Spectral Difference Occurrences processed from the 
    spectral images using the given spatial distance and direction.
    As the difference is defined using the KLPD, A two dimensional representation 
    is obtained (shape diff, intensity diff).
    
    To obtain the occurrence matrix (so an histogram), the np.2Dhist() must be applied.
    Take care about the selected number of bins (sturges rules for example) to avoir bias.
    
    This version produced a matrix defined in the quarter plane of the positive 
    measures. 
    
    Arguments:
        SpecImg : 
            the 3-dimensional np.array corresponding to the spectral image
            
        Spatial_dist :
            a np.integer corresponding to the norm of the spatial vector for the 
            cooccurrence processing. When direction is equal to 45 or 135°, the 
            the spatial distance is considered as the radius of the circle and divide 
            by sart(2.0) for the horizontal and vertical displacement.
            
        Direction :
            a np.integer corresponding to a given direction (0°, 45°, 90°, 135°).
            
        Resolution :
            a np.float32 corresponding to the spectral resolution of the HS image
            
            
        Returns:
            SDO : the 2 dimensional np.array corresponding to the array of 
                   spectral difference occurrences.
                 be carefull : SDO is the distribution of differences, not an histogram !
            
    '''

    [NbLg, NbCol, NbWaves] = SpecImg.shape
    
    if (Direction ==0 ):
        Col_offset = Spatial_dist          ## Vector displacement
        Lg_offset =0
        Col_start = 0                      ## starting corner
        Lg_start = 0
        Col_final  = NbCol - Spatial_dist  ##Final corner
        Lg_final  = NbLg
        
    elif (Direction == 45):
        Col_offset = np.around(Spatial_dist / np.sqrt(2.0)) 
        Lg_offset = np.around(Spatial_dist / np.sqrt(2.0))
        Col_start = 0                      
        Lg_start = 0
        Col_final  = np.around(NbCol - Spatial_dist / np.sqrt(2.0)) 
        Lg_final  = np.around(NbCol - Spatial_dist / np.sqrt(2.0)) 
        
    elif (Direction == 90):
        Col_start = 0
        Lg_start = Spatial_dist
        Col_final  = NbCol
        Lg_final  = NbLg - Spatial_dist

    elif (Direction == 135):
        Col_offset = - np.around(Spatial_dist / np.sqrt(2.0)) 
        Lg_offset = - np.around(Spatial_dist / np.sqrt(2.0))
        Col_start = Spatial_dist / np.sqrt(2.0)
        Lg_start = Spatial_dist / np.sqrt(2.0)
        Col_final  = NbCol
        Lg_final  = NbLg 
        

    else:
        print("Proposed spatial direction is not managed, use 0,45,90 or 135 !")
        return -1
            
    ## Process the cooccurrence between th original image and the sliced one
    SDO = HSI_dist.pseudo_div_KL2(SpecImg[Lg_start:Lg_final , Col_start:Col_final, :].reshape( ((Lg_final-Lg_start)*(Col_final-Col_start),NbWaves)) ,
                                   SpecImg[(Lg_start+Lg_offset):(Lg_final+Lg_offset) , (Col_start+Col_offset):(Col_final+Col_offset), :].reshape( ((Lg_final-Lg_start)*(Col_final-Col_start),NbWaves)) ,
                                   Resolution)

    return SDO        


################################################################################
################################################################################
#####                                                                       ####
#####     Hurst_Processing : Hurst coefficient processing from an HSI       ####
#####                                                                       ####
################################################################################
################################################################################
def  Ellipse_2D(Tab_2D, Control_plot = False):
    '''
    Modelize a distribution of 2D coordinates as a Normal law, so an ellipse.
    Return the coordinate of the ellipse : center, main and second vector, 
    and the main and second radius corresponding to the eigen values.
    
    99.7% of the pixels are included inside a radius = 3 sigma

    Parameters
    ----------
    Tab_2D : np.array( (nb_samples,2) )
        An array of nb_samples having 2 coordinates x and y

    Returns
    -------
    Avg : np.array( 2 )
         2D coordinates of the distribution center 
    
    Vec : np.array( (2,2) )
        A table embedding the coordinates of the eigen vectors
    
    Radius : np.array(2)
        The radius of the equivalent ellipsoid at 1 sigma. 99.7% of data are
        inside a radius of 3 sigma
        

    '''
    # Statistical data
    estimated_avg = np.average(Tab_2D, axis=0)
    estimated_cov_mat = np.cov(Tab_2D.transpose())

    # Extracting the Eigen Vectors
    eigen_val, eigen_vector = np.linalg.eig(estimated_cov_mat)
    sort_indices = np.argsort(eigen_val)[::-1]  
    Main_Eigen_vector , Second_Eigen_vector = eigen_vector[:, sort_indices]
    
    # Extracting the parameters for the ellipse : radius and angle
    eigen_val = np.sqrt(eigen_val)
    estimated_theta = np.rad2deg(np.arctan(Main_Eigen_vector[1]/ Main_Eigen_vector[0]))

    
    # Plotting the Eigen Vector for control (Main eigen vector in red, with biggest length)
    if (Control_plot == True):
        plt.figure()
        ax = plt.subplot(111)
        # plt.xlim([-10.0, 10.0])
        # plt.ylim([-10.0, 10.0])
    
        # ax.scatter(Tab_2D[:,0], Tab_2D[:,1], marker='o', color='b', alpha=0.5)
        ax.hist2d(Tab_2D[:,0], Tab_2D[:,1], bins = 75, cmin = 2)

        length = 2.0
        plt.plot([Main_Eigen_vector[0]*2*(-length) +estimated_avg[0], Main_Eigen_vector[0]*2*length +estimated_avg[0]], 
                  [Main_Eigen_vector[1]*2*(-length) +estimated_avg[1], Main_Eigen_vector[1]*2*length +estimated_avg[1]], 
                  color ='r')
        
        plt.plot([Second_Eigen_vector[0]*(-length) +estimated_avg[0], Second_Eigen_vector[0]*length +estimated_avg[0]], 
                  [Second_Eigen_vector[1]*(-length)+estimated_avg[1] , Second_Eigen_vector[1]*length +estimated_avg[1]], 
                  color ='g')
    
        # Plot the Ellipse at 1, 2 and 3 Sigma    
        for j in range(1, 4):
            ell = Ellipse(xy=(estimated_avg[0], estimated_avg[1]),
                          width=eigen_val.max()*j*2, height=eigen_val.min()*j*2,
                          angle= estimated_theta  )
            ell.set_facecolor('none')
            ell.set_alpha(0.2)
            ell.set_color('b')
            ell.set_edgecolor('r')
            
            ax.add_artist(ell)
        plt.show()

    Vect = np.zeros( (2,2) )
    Vect[0,:] = Main_Eigen_vector
    Vect[1,:] = Second_Eigen_vector
    
    return estimated_avg, Vect, eigen_val
################################################################################
################################################################################
#####                                                                       ####
#####     Hurst_Processing : Hurst coefficient processing from an HSI       ####
#####                                                                       ####
################################################################################
################################################################################

def Hurst_Processing(ImgTab, ImgType, SpatialRange, Direction, Wavelengths, Mode, diffeomorphism):
    '''
    Process the evolution of the Area of the Image Differences for a set of spatial distances
    and a given orientation. The Hurst coefficient is defined thanks to equation of the 
    fractional Brownian movement. According to the particularities of the used pixel value
    difference, a diffeomorphism can be required (SDOM case with the KLPD).

    Parameters
    ----------
    ImgTab : np.array 
        a 2d-array for intensity image or nD-array for colour/multi/hyperspectral or multivariate image
        
    ImgType : string : 'hdr', 'colour', 'multispec' 
        Type of the input image Hyperspectral ('hdr'), colour or multispectral.
        
    SpatialRange : 1D-array 
        The spatial distances to use for the spatial displacement to apply for the Image value difference processing.
        
    Direction : np.integer 
        Corresponde to the selected direction  0°, 45°, 90° or 135° for the SDOM processing.
        if Direction = -1, the SDOM is processed for the 4 directions and summed (isotropic measure)
        
    Wavelengths : 1D-array
        List of wavelength for the hyperspectral case.
        
    Mode : string : 'direct', 'ellipse'
        Define how must be processed the value to return
            direct  : process the square average of the total spectral difference
            ellipse : process the ellipse embedding 99.7% of the samples
            entropy : process the entropy of the BHSD
                
    Diffeomorphism : string : '', 'Log', 
        Define the lack or the type of the applied diffeomorphisme to the area/volume measures
        before the slope processing. REquired to obtain a distribution closer to a Normal law.

    Returns
    -------
    H : np.float
        the processed Hurst coefficient for the proposed image using the spatial range and spatial orientation
    
    Vol : np.array of float
        the array of area/volumes of the pixel value differences (for post-processing/display, ...)
        
    offset : np.float
        the offset for x=0 of the identified straight line

    '''

    ## create the local variables
    Vol        = np.zeros( SpatialRange.size)
    Resolution = Wavelengths[1] - Wavelengths[0]
    
    ## Step 1 : For each spatial distance : process the area/volume of the 
    ##          histogram of pixel image differences, then the area of the
    ##          distribution using an ellipsoid.
    ##
    for i in range(SpatialRange.size):
        if (Direction == -1) : ## isotropic case
            SDOM_matrix = SDOM(ImgTab, SpatialRange[i], 0, Resolution)
            SDOM_matrix = SDOM_matrix + SDOM(ImgTab, SpatialRange[i], 45, Resolution)
            SDOM_matrix = SDOM_matrix + SDOM(ImgTab, SpatialRange[i], 90, Resolution)
            SDOM_matrix = SDOM_matrix + SDOM(ImgTab, SpatialRange[i], 135, Resolution)
            SDOM_matrix = SDOM_matrix/4
        else :
            SDOM_matrix = SDOM(ImgTab, SpatialRange[i], Direction, Resolution)
            
            
        ## Step 2 : Apply the diffeomorphism    
        if (diffeomorphism =='Log'):
                SDOM_matrix = np.where(SDOM_matrix == 0 , 1E-3, SDOM_matrix)
                SDOM_matrix = np.log(SDOM_matrix)
                
        elif (diffeomorphism != 'No'):
            print('ERROR ######################################')
            print('ERROR ######################################')
            print('ERROR ######################################')
            print('requested diffeomorphism for Hurst processing not supported')
            print('ERROR ######################################')
            print('ERROR ######################################')
       
      

        ## Step 3 : Process the Area / Volume of the difference distribution
        if (Mode == "ellipse"):
            Avg, Vect, Radius = Ellipse_2D(SDOM_matrix, Control_plot = True)
            Vol[i] = 3 * Radius[0] *Radius[1] ## 3 sigma = 99.7% of data, keep only the variable part, not the constants
        
        elif (Mode == 'direct'):
            Vol[i] = np.mean( SDOM_matrix.sum(axis=1)**2 )
        
        elif (Mode == 'entropy'):
            histo = np.histogram(SDOM_matrix.sum(axis=1), bins=100, range=(0, 1E3), density=True)
            ##data  = histo[0]
            ##entropy = - (data * np.log(np.abs(data))).sum()
            entropy = stats.entropy(histo[0])
            
            ## For normal distribution, Entropy = ln(\sigma . \sqrt{2 .\pi . e} )
            Vol[i] = np.exp(entropy)

        ## end for i in range(SpatialRange.size)
        ##--------------------------------------
        
    ## Step 4 :Approximation of the area/Vol evolution as a straight line 
    ##         we need only the slope, not the offset
    
    H, offset = np.polyfit(np.log(SpatialRange), np.log(Vol), 1)
    
    return H, Vol, offset
        
        