# -*- coding: utf-8 -*-
"""
Created on Tue Mar 07 14:30:02 2017
    HSI_library.py
    
    Purpose : list of Generic function for HSI images (Pavia, Indian_Pines, ...)
              constants are in HSI_remote_sensing_scene_datas.py

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
        plot_Spectral_Difference_Map: 2D maps (using alpha transparency value)
        plot_3D_Spectral_Difference_Map: plot the 3D maps on the current figure
        plot_BHSD
    
    status : Ok

    date   : 2018/03/26, validated again the 2020/05/30
    
    Notes : To change / adapt the resolution variable and to replace by the wavelength list
            for non-regular spectral sampling.
    
@author: nrichard
"""
import numpy                             as np
import scipy                             as sp
import matplotlib                        as mpl
from matplotlib            import pyplot as plt  ## for the main function
from matplotlib.colors     import LogNorm        ## for BHSD plot
#from matplotlib.patches import Ellipse

import spectral                          as spy
import spectral.io.envi                  as envi       ## for the spectral image load, save and plot

#import HSI_remote_sensing_scene_datas    as HSI_data    # for required for Pavia Processing and open_mat_Spec_img

################################################################################
################################################################################
################################################################################

##=============================================================================
## Fonctions locales
##=============================================================================
def open_mat_Spec_img(ImgNumber, Corrected=False, Range ='Whole'):
    '''
        Return the spectral and groundtruth image corresponding to the specified Image file
        the corresponding file is a matlab one requiring a specific reading protocol
        
        Arguments:
            ImgNumber : Image number in the list of the spectral images (from HSI_remote_sensing_scene_datas.py)
            Corrected : Boolean to indicate the direct data or the corrected one
            Range     : 'Whole', 'Vis', 'IR', 
        
        return :
            Img      :   the spectral image as an array of spectra (nblg x nbcol x nb_wavelengths) of double real
            GroundTruthImg  : the associated ground-truth image(nblg x nbcol ) of uint
            Wavelist : the corresponding wavelength list
            Resolution : the spectral resolution (for integration purpose)
    '''
    ## Extract the Spectral image and Remove null values in the reflectance spectrum
    if (Corrected == True):
        img_mat = sp.io.loadmat("data\\" + HSI_data.filename_corrected[ImgNumber] + ".mat")   
        Img = img_mat[ HSI_data.filename_corrected[ImgNumber]  ]                   
    else:
        img_mat = sp.io.loadmat("data\\" + HSI_data.filename[ImgNumber] + ".mat")   
        Img = img_mat[ HSI_data.filename[ImgNumber]  ]                   
        
    Img = np.where( Img < 1, 1, Img)


    ## Extract the Ground-truth image
    img_mat = sp.io.loadmat("data\\" + HSI_data.filename[ImgNumber] + "_gt.mat")   
    GTImg = img_mat[ HSI_data.filename[ImgNumber]+"_gt" ]  
    if (ImgNumber == 4):
        ## Post processing of the Ground-Truth image for the SalinasA case (15 labels, but 6+1 used)
        GTImg[ GTImg == 10] = 2
        GTImg[ GTImg == 11] = 3
        GTImg[ GTImg == 12] = 4
        GTImg[ GTImg == 13] = 5
        GTImg[ GTImg == 14] = 6
        

    ## process the corresponding Wavelength List
    resolution = np.float(HSI_data.MaxWave[ImgNumber] - HSI_data.MinWave[ImgNumber])/HSI_data.BandCount[ImgNumber]    
    Wavelist = np.arange(HSI_data.MinWave[ImgNumber], HSI_data.MaxWave[ImgNumber] +1 , resolution)
    if (Wavelist.size != Img[0,0,:].size):
        Wavelist = np.arange(HSI_data.MinWave[ImgNumber], HSI_data.MaxWave[ImgNumber], resolution)
    
    ## reduce the dataset to the selected range
    if (Range == 'Vis') :
        ## search the first index in the list with wavelength >= 360.0
        if ( (Wavelist[0] >= HSI_data.Min_Vis)  ):
            FirstWaveIndex = 0
        else: ## case (Wavelist[0] < 360.0):
            listWaves, = np.where( Wavelist >= HSI_data.Min_Vis)
            FirstWaveIndex = listWaves[0]-1

        if (Wavelist.max() >= HSI_data.Max_Vis):
            listWaves, = np.where( Wavelist >= HSI_data.Max_Vis )
            LastWaveIndex = listWaves[0]
        elif (Wavelist.max() < HSI_data.Max_Vis ):
            LastWaveIndex = Wavelist.size -1                    

        return [Img[:, :, FirstWaveIndex:LastWaveIndex], GTImg, Wavelist[FirstWaveIndex:LastWaveIndex], resolution]
    
    elif (Range == 'IR'):
        ## search the first index in the list with wavelength >= 360.0
        if ( Wavelist[0] >= HSI_data.Min_IR):
            FirstWaveIndex = 0
        elif (Wavelist.max() < HSI_data.Min_IR):
            print ('No IR data in this image')
            return []
        else: ## case (Wavelist[0] < 360.0):
            print ('Standard case : spectral image with Vis and IR parts')
            listWaves, = np.where( Wavelist >= HSI_data.Min_IR)
            FirstWaveIndex = listWaves[0]-1
            
        LastWaveIndex=Wavelist.size-1
        print ('Retained index for the study : first index ='+ str(FirstWaveIndex) + 'last index = '+ str(LastWaveIndex))
        return [Img[:, :, FirstWaveIndex:LastWaveIndex], GTImg[:, :], Wavelist[FirstWaveIndex:LastWaveIndex], resolution]        
    else:
        return [Img, GTImg, Wavelist, resolution]
##=============================================================================
def open_envi_image(filename, Range ='Whole'):
    '''
       load a spectral image and return an array of spectra

       Arguments:
           filename  :  name of the envi spectral image file to open
           Range     : 'Whole', 'Vis', 'IR',  : not implemented !!!

       Return:
           img        : the  Spy structure for hyperspectral img
           SpecImg    : the spectral image as an array of spectra (nblg x nbcol x nb_wavelengths) of double real
           Wavelist   : the corresponding wavelength list
           Resolution : the spectral resolution (for integration purpose) 

    '''
    ## Step 1 : read the hdr file (text file) to read the spectral image parameters
    try:
        img                  = spy.open_image(filename)
    except IOError:
        print('File ' + filename + 'NOT found')
        return [-1, -1, -1, -1]
    
#    NbLg, NbCol, NbWaves = img.nrows, img.ncols, img.nbands
    Wavelist             = np.array(img.bands.centers)
    Resolution           = Wavelist[1]-Wavelist[0]
    
    ## Step 2 : extract the spectral array : nblg x nbcol x nbwaves
    SpecImg= np.array(img.load(), np.float32)     ## for the following, require a np.array

    return [img, SpecImg, Wavelist, Resolution] 
    ## Non fini
##=============================================================================
def save_ENVI_Img( SpecImg, DictData, filename):
    '''
        save a 3D array as a spectral image in the ENVI mode associated to a 
                                            text data file with hdr extension.
        
        Arguments:
            SpecImg : a numpy 3D array tab : NbLig x NbCol x NbWaves
            
            DictData : Dictionnary of Data associated to the spectral image
                       (DictImgData = SpyImg.metadata.copy() for example, SpyImg obtained using open_envi_image)
                       or
                           DictData ={ 'lines'   : nb_lg,
                                        'samples' : nb_col,
                                        'bands'   : nb_waves,
                                        'data type': 5, ##5 =dble precision, 4= float, 2= 16-bit singed integer, 12 = uint16
                                        'wavelength units': 'nm'
                                    }    
                           DictData['wavelength']=wavelengthes
            
            filename : a string with the full file noame
            
        Return:
                0 if OK, -1 if problems
    '''
    ## spy.envi.save_image update the dictionary with the right number of lines and columns
    
    spy.envi.save_image(filename, SpecImg, metadata = DictData, force = True, dtype = np.float32)
    
##=============================================================================
def Process_Reflectance_linear(Img_FileName, Black_FileName, White_FileName):
    '''
        Reflectance processing for a linear sensor 
           (spectralon is acquired in the bottom of the image, not the same image height)
        Return a reflectance image from the Spectral Img, and black & white images.
        The reflectance = (SpyImg - SpyBlack) / (SpyWhite - SpyBlack)
        Use the filename to avoid to transfert data of big size
        
        Arguments:
            Img_FileName        : the Spectral image 
            Black_FileName        : the Spectral black image 
            White_FileName        : the Spectral white image 
            
        return :
            SpyImg          : the Spy structure of the radiance image, including the dictionnary
            Reflectance     : the processed spectral array of reflectances
            Waves           : the Wavelength array
            Resolution      : the spectral resolution of the initial image
        
    '''
    ##=========================================================================
    ##========== Step 1 : read the Black Img                            =======
    ##=========================================================================
    ## Read the black img, process a line of average black value
    [SpyBlackImg, BlackImg, Waves, resolution] = open_envi_image(Black_FileName, 'Whole' )
    Black_weight = BlackImg.min(axis=0) ## extrait le min pour éviter des valeurs négatives
    
    ##=========================================================================
    ##========== Step 2 : read the White Img                            =======
    ##=========================================================================
    ## Read the White img, process a line of average white value
    [SpyWhiteImg, WhiteImg, Waves, resolution] = open_envi_image(White_FileName, 'Whole' )
    White_weight = WhiteImg.max(axis=0) ## extrait le max pour éviter des valeurs négatives

    ##=========================================================================
    ##========== Step 3 : read the Spectral Img                         =======
    ##=========================================================================
    ## Read the Spectral img, and normalise each line using the black and white references
    [SpyImg, SpecImg, Waves, resolution] = open_envi_image(Img_FileName, 'Whole' )
 
    Reflectance = np.zeros_like(SpecImg)
    ## replace the destroyed pixels by their neighbour
    pos = np.where((White_weight - Black_weight) == 0)
    for i in range(pos[0].shape[0]):
        col  = pos[0][i]
        w    = pos[1][i]
        White_weight[col, w] = White_weight[col+1, w]
        Black_weight[col, w] = Black_weight[col+1, w]
    # Process the reflectance    
    for i in range(SpecImg.shape[0]):
        Reflectance[i,:,:] = np.float32(10**(-32)) + (SpecImg[i,:,:] - Black_weight)*0.9999 / (White_weight - Black_weight)
    
    return SpyImg, Reflectance, Waves, resolution
##=============================================================================
	
##=============================================================================
def Process_Reflectance_2D_snapshot(Img_FileName, Black_FileName, White_FileName):
    '''
        Reflectance processing for a snapshot sensor
        (white & black images of the same size than the Img are acquired )
        Return a reflectance image from the Spectral Img, and black & white images.
        The reflectance = (SpyImg - SpyBlack) / (SpyWhite - SpyBlack)
        Use the filename to avoid to transfert data of big size
        
        Arguments:
            Img_FileName        : the Spectral image 
            Black_FileName        : the Spectral black image 
            White_FileName        : the Spectral white image 
            
        return :
            SpyImg          : the Spy structure of the radiance image, including the dictionnary
            Reflectance     : the processed spectral array of reflectances
            Waves           : the Wavelength array
            Resolution      : the spectral resolution of the initial image
        
    '''
    ##=========================================================================
    ##========== Step 1 : read the Black Img                            =======
    ##=========================================================================
    ## Read the black img, process a line of average black value
    [SpyBlackImg, BlackImg, Waves, resolution] = open_envi_image(Black_FileName, 'Whole' )
    
    ##=========================================================================
    ##========== Step 2 : read the White Img                            =======
    ##=========================================================================
    ## Read the White img, process a line of average white value
    [SpyWhiteImg, WhiteImg, Waves, resolution] = open_envi_image(White_FileName, 'Whole' )

    ##=========================================================================
    ##========== Step 3 : read the Spectral Img                         =======
    ##=========================================================================
    ## Read the Spectral img, and normalise each line using the black and white references
    [SpyImg, SpecImg, Waves, resolution] = open_envi_image(Img_FileName, 'Whole' )
 
    Reflectance = np.zeros_like(SpecImg)
    # Process the reflectance    
    Reflectance = np.float32(10**(-32)) + (SpecImg - BlackImg)*0.9999 / (WhiteImg - BlackImg)
    
    return SpyImg, Reflectance, Waves, resolution
##=============================================================================

##=============================================================================
def Spec2ENVI_RGB(SpecImg, Wavelengths, Normalisation = 'Marginal'):
#def Spec2ENVI_RGB(SpecImg, ImgNumber, resolution):
    '''
        Return a trichromatic image (Envi mode) for imshow() plotting
        
        Arguments:
            SpecImg   : the Spectral image          
            Waves     : wavelenghtes corresponding to the spectral data
            Normalisation : "Marginal" (channel separately) or "Global" (preserving the relative contrasts)
			
        return :
            RgbImg         : the RGB image for plotting
        
    '''
    ## image size
    
    [NbLg, NbCol, NbWaves] = SpecImg.shape

    # affichage de contrôle par sélection de canaux RGB
    resolution = (Wavelengths[NbWaves-1]-Wavelengths[0])/NbWaves
    canal_B = np.uint((470-Wavelengths[0]) / resolution)
    canal_G = np.uint((520-Wavelengths[0]) / resolution )
    canal_R = np.uint((650-Wavelengths[0]) / resolution)
    
    img_RGB = np.zeros( (NbLg, NbCol, 3), dtype= float)

    # RGB values reduced between 0 to 1 for displaying (global normalisation to avoid marginal artefacts)
    img_RGB[:, : ,0] =  np.float32(SpecImg[:,:, canal_R])
    img_RGB[:, : ,1] =  np.float32(SpecImg[:,:, canal_G])
    img_RGB[:, : ,2] =  np.float32(SpecImg[:,:, canal_B])
    
    if (Normalisation != 'Marginal'):
    ## Global Stretching
        maxi  = np.float32(img_RGB.max())
        mini  = np.float32(img_RGB.min())
        img_RGB = (img_RGB - mini) / (maxi-mini)
    else:
    ## Local Stretching
        for i in range(3):
            img_RGB[:,:,i] = (img_RGB[:,:,i]-img_RGB[:,:,i].min()) / (img_RGB[:,:,i].max()-img_RGB[:,:,i].min())        

#    Gamma = 0.8
#    img_RGB = img_RGB**(1/Gamma)
    return img_RGB
##=============================================================================


##=============================================================================
def Spec2CMF_XYZ(SpecImg, Waves, reflectance = False, ligthning='D65')  :
    '''
        create an XYZ image using the CIE-XYZ values 
        require to interpolate the CIE-XYZ 2° and Illuminant data to adapt them to the sensor data
        
        Visible rangle = 360 -780 nm, reduced to MinWaveSensor - 780 nm (Adaptation to Pavia and other HSI images)

        Parameters:
        ===========
            :param SpecImg: the Spectral Image
            :type SpecImg: float numpy array of NbLG x NbCol x NbWaves

            :param Waves : table of wavelenths from the hyperspectral image
            :type Waves: float numpy array of NbWaves

            :param reflectance: True for reflectance Spectra
            :type refelctance: boolean (True or False)

            :param lightning : nature of the illuminant to use for reflectance case, D50 is used if not specified or don't owning to D50, D65, E
            :type lightning: D50, D65 or E

        Returns:
        ========
            :return: XYZImg with values in [0.0 , 1.0]
            :rtype: numpy.array (nbLg, nbCol, 3)

    '''
    (lg, col, wavecount) = SpecImg.shape

    if ((ligthning != 'D65') and (ligthning != 'D50') and (ligthning != 'E')):
        ligthning = 'D50'
        print('Error in ligthning specification, automatic selection = D50')
        
    ## Identify the CIE and Ill wavelengthes to min and max of SpecImg. Adjust the range to it in the limit [360, 780] nm
    if (Waves[0] > 360.0):
        First_CIE_wave = ((Waves[0] - 360.0) / 5.0).astype(int) #The lowest wavelength of SpecImg > 360 nm
        First_SpecImg_wave = 0
#    elif (Waves[0] < 360.0):
#        First_CIE_wave = 0
#        First_SpecImg_wave = 0 #### To finish for a generic function  !!!!!
    else:
        First_CIE_wave = 0
    
    ## upper limits for the interpolation and then integrals
    if (Waves.max() > 780.0):
        tmp_wave, = np.where( Waves>780.0 )
        Last_SpecImg_wave = tmp_wave[0]
        Last_CIE_wave = 85              ## 84 correspond to 780.0
        
    elif (Waves.max() == 780.0):
        Last_SpecImg_wave = Waves.size   ##replace  Waves.size -1, the value correspond to the first index after the last one to keep
        Last_CIE_wave = 85 

    else:
        tmp_wave, = np.where( Waves == Waves.max() )
        Last_SpecImg_wave = tmp_wave[0]+1  ## replace tmp_wave[0], the value must be the nb of samples
        Last_CIE_wave = (np.ceil((Waves.max() -360.0)/5.0)).astype(int) ## must choose the upper integer in the list for the interpolation 
    

    ## Read the CIE-XYZ matching function (2°), from 360 to 830nm, step = 5nm
    CMF_X = np.load('data\\X_360_830_2deg.npy')[First_CIE_wave:Last_CIE_wave] ## X = np.load('data\\x_360_830_2deg.npy')[8,-10] # to reduce to 400:790
    CMF_Y = np.load('data\\Y_360_830_2deg.npy')[First_CIE_wave:Last_CIE_wave]
    CMF_Z = np.load('data\\Z_360_830_2deg.npy')[First_CIE_wave:Last_CIE_wave]

    ## Read the light spectrum (from 300 to 780nm, step = 5nm)
    light_Spectrum = np.load('data\\I_'+ligthning+'_300-780.npy')[(12+First_CIE_wave):12+Last_CIE_wave]
            
    ## interpolate the CIE and lightning curves to be adapted to the spectral sensor
    Waves_CIE    = np.arange(360.0 + First_CIE_wave*5.0 , 361.0+(Last_CIE_wave-1)*5.0 ,5) ## the measurement point for the CIE curves
    
    ## ! ## Potentially sp.interp is depreciated and should be replaced by np.interp //ERROR
    CMF_X_interp = sp.interp(Waves[0:Last_SpecImg_wave],Waves_CIE,CMF_X)
    CMF_Y_interp = sp.interp(Waves[0:Last_SpecImg_wave],Waves_CIE,CMF_Y)
    CMF_Z_interp = sp.interp(Waves[0:Last_SpecImg_wave],Waves_CIE,CMF_Z)
    light_Spectrum_interp   = sp.interp(Waves[0:Last_SpecImg_wave],Waves_CIE,light_Spectrum)
    wavecount = light_Spectrum_interp.size

    ## Transform the spectral image  in a table of lg x col spectra
    Spectral_list = np.matrix( SpecImg[:,:, 0:Last_SpecImg_wave].reshape(lg * col, wavecount) )

    ## Compute the XYZ image (bruce lindbloom equations)
    if (reflectance == True)    :
        mat = np.matrix( [CMF_X_interp, CMF_Y_interp, CMF_Z_interp] * light_Spectrum_interp )
        XYZ_list = Spectral_list * mat.transpose()
        XYZ_list = XYZ_list / np.dot(CMF_Y_interp,light_Spectrum_interp)
    else: ## Partie validée NR: 15/10/2016
        mat = np.matrix( [CMF_X_interp, CMF_Y_interp, CMF_Z_interp]  )
        XYZ_list = Spectral_list * mat.transpose()
        XYZ_list = XYZ_list / np.sum(CMF_Y_interp)

    # As a normalisation is applied inside the RGB transformation, no need at this step
    return np.array(XYZ_list).reshape( (lg,col,3))
##=============================================================================
    
def XYZ_2_RGB(ImgXYZ, RGB_space ='sRGB', Gamma=2.2):
    '''
        create an RGB image using the CIE-XYZ values 
        require to select an RGB space (sRGB, Adobe, CIE, ....)

        Parameters:
        ===========
            :param ImgXYZ: the Image in the primary space CIE XYZ (2°)
            :type  ImgXYZ: float numpy array of NbLG x NbCol x 2

            :param RGB_space : selected RGB space for the transform 
            :type RGB_space  : sRGB(D65), AdobeRGB(D65), other to define...

            :param Gamma : Gamma value for the Non-linear visual adaptation (Gamma approx 2.2 for s-RGB), more important is Gamma more white is the image, less important more darky
            :type Gamma : numpy float
        Returns:
        ========
            :return: RgbImg with values in [0.0 , 1.0]
            :rtype : float numpy.array (nbLg, nbCol, 3)

    '''
    (lg, col, poub) = ImgXYZ.shape
    XYZ_list = ImgXYZ.reshape(lg*col,3)
    
    if (RGB_space =='AdobeRGB'): ## AdobeRGB - D65 , Bruce Lindbloom website
        Mm1 = np.matrix (   [ [2.0413690  , -0.5649464 , -0.3446944 ],
                              [-0.9692660 ,  1.8760108 ,  0.0415560 ],
                              [0.0134474  , -0.1183897 , 1.0154096  ] ]
                        )
    elif (RGB_space =='sRGB'): ## sRGB - D65 , Bruce Lindbloom website
        Mm1 = np.matrix (   [ [3.2404542 , -1.5371385 , -0.4985314 ],
                              [-0.9692660,  1.8760108 , 0.0415560],
                              [ 0.0556434, -0.2040259 , 1.0572252  ] ]
                        )
    else:
        print("the other matrix are not yet implemented !")
        
    ImgRGB = XYZ_list * Mm1.transpose()

    ## ramène les valeurs RGB dans la dynamique globale [0,1] : validée NR 15/10/2016
    mini = ImgRGB.min() ## attention : ne pas normaliser de faàon marginal !!!!
    maxi = ImgRGB.max()
    ImgRGB= (ImgRGB - mini)/(maxi-mini)

    ## transforme la liste RGB en image lg x col x 3 canaux (R, G, B)
    ImgRGB = np.array(ImgRGB).reshape(lg,col,3)

    ## transformation Gamma pour ramener la dynamique du sRGB (Gamme =2.2 pour D65)
    ImgRGB= ImgRGB**(1/Gamma)

    return ImgRGB

##=============================================================================
def Spec2CMF_RGB(SpecImg, Waves, reflectance = False, ligthning='D65', RGB_space='sRGB', Gamma=2.2)  :
    '''
        create an RGB image using the CIE-RGB values (NR dec2016- Ok)
        require to interpolate the CIE-XYZ 2° and Illuminant data to adapt them to the sensor data
        
        Visible rangle = 360 -780 nm, reduced to MinWaveSensor - 780 nm (Adaptation to Pavia and other HSI images)

        Parameters:
        ===========
            :param SpecImg: the Spectral Image
            :type SpecImg: float numpy array of NbLG x NbCol x NbWaves

            :param Waves : table of wavelenths from the hyperspectral image
            :type Waves: float numpy array of NbWaves

            :param reflectance: True for reflectance Spectra
            :type refelctance: boolean (True or False)

            :param lightning : nature of the illuminant to use for reflectance case, D50 is used if not specified or don't owning to D50, D65, E
            :type lightning: D50, D65 or E

            :param Gamma : Gamma value for the Non-linear visual adaptation (Gamma approx 2.2 for s-RGB), more important is Gamma more white is the image, less important more darky
            :type Gamma : numpy float
        Returns:
        ========
            :return: RgbImg with values in [0.0 , 1.0]
            :rtype: numpy.array (nbLg, nbCol, 3)

    '''
    ImgXYZ = Spec2CMF_XYZ(SpecImg, Waves, reflectance, ligthning)
    ImgRGB = XYZ_2_RGB(ImgXYZ, RGB_space, Gamma)
    return ImgRGB
#
###=============================================================================
def Spec2RGB_MF(SpecImg, Waves, CMF_Set):
    '''
        Convert a hyperspectral image into a RGB image using a set of 3 Colour Matching Functions
        proposed as argument for the same spectral range defined by the wavelengths.
        Require 3 arrays corresponding to the values of the CMF for the corresponding 
        wavelengths of the Spectral Image or List.
        
        The CMF can be created using the function or main of HSI_Create_Colour_Matching_Functions.py
        
        2 images are returned : one preserving the relative contrast between the 3 channels, 
                                and one with magnified constrats.
        
        Parameters:
        ===========
            :param SpecImg: the Spectral Image or Spectral List
            :type  Specimg: float numpy array  of (NbLG x NbCol x NbWaves) or (NbSamples x NbWaves)

            :param Waves : table of wavelenths from the hyperspectral image
            :type Waves: float numpy array of NbWaves
            
            :param CMF_Set: An array of 3 spectral functions processed for Wavelengths
            :type  CMF_Set: float numpy array of 3 x NbWaves
        Returns:
        ========
            :return: [RgbImg, RgbImg_M] with values in [0 , 255]
            :rtype: numpy.array of uint8 and size (nbLg, nbCol, 3)
    '''
    if (SpecImg.ndim == 2):
        ## SpecImg = list of spectra of size (NbLg xNbCol, NbWaves)
        [ NbSpectra , NbWaves ] = SpecImg.shape 
        SpecList = SpecImg
        
    elif (SpecImg.ndim == 3):
        ## SpecImg = spectral Image of size (NbLg, NbCol, NbWaves)
        [ NbLg, NbCol , NbWaves ] = SpecImg.shape 
        NbSpectra = NbLg*NbCol
        SpecList = SpecImg.reshape( NbSpectra, NbWaves )

    elif (SpecImg.ndim == 1 ):
        ## Only 1 spectra
        SpecList = SpecImg.reshape( 1, NbWaves )
    else:
        print("Dimension of the Spectral Image not adapted to Spec2RGB_MF")
        return

    ## Applied the CMF to the spectral image/list
    RGB_List     = np.zeros( (NbSpectra, 3), dtype = np.float32 )
    RGB_List_Mag = np.zeros( (NbSpectra, 3), dtype = np.float32 )
    
    for j in range(NbSpectra):   
        RGB_List[j,2]= np.dot(CMF_Set[0,:] , SpecList[j,:]) ## Blue channel
        RGB_List[j,1]= np.dot(CMF_Set[1,:] , SpecList[j,:]) ## Green channel
        RGB_List[j,0]= np.dot(CMF_Set[2,:] , SpecList[j,:]) ## Red channel
                                    
    ## Normalisation = balance des blancs, maintien de la correspondance couleur
    mini = RGB_List.min()
    maxi = RGB_List.max()
    RGB_List = 255.0*(RGB_List-mini)/(maxi-mini) 
    
    ## Normalisation avec accroissement contraste (perte contraste relatif entre canaux)
    for i in range(3):
        mini = RGB_List[:,i].min()
        maxi = RGB_List[:,i].max()
        RGB_List_Mag[:,i] = 255.0*(RGB_List[:,i] - mini)/(maxi-mini)
        
    RGB_Img = RGB_List.astype(np.uint8)
    RGB_Img_Mag = RGB_List_Mag.astype(np.uint8)

    if (SpecImg.ndim ==  3):
        ## SpecImg = list of spectra of size (NbLg xNbCol, NbWaves)
        RGB_Img = RGB_Img.reshape( (NbLg, NbCol, 3) )
        RGB_Img_Mag = RGB_Img_Mag.reshape( (NbLg, NbCol, 3) )

    return [RGB_Img, RGB_Img_Mag]
###=============================================================================

##=============================================================================
#def plot_3Dellipsoid(Avg, Var_Covar, ax):
#    '''
#    plot a 3D-ellipsoïd in the current figure
#    used reference : https://kittipatkampa.wordpress.com/2011/08/04/plot-3d-ellipsoid/ (matlab source but with some mistakes !)
#    
#    Arguments:
#        Avg: 
#            1x3 array with the average position of the ellipsoïd in a 3D space
#        
#        Var_CoVar:
#            3x3 matrix of Variance-Covariance
#        
#        ax :
#            subplot for a 3D representation ("ax = fig.add_subplot(111, projection='3d')"  )
#    Returns: 
#        Nothing
#    '''
#    [EigenValues, EigenVectors] = np.linalg.eig(Var_Covar) ##U : eigenvalues (1x3) , V (3x3) : eigenvectors in columns
#    scale = 3.0
#    radius = scale * np.sqrt(EigenValues) # (1x3) 
#    
#    ## sample a sphere 
#    u = np.linspace(0.0, 2.0 * np.pi, 100)
#    v = np.linspace(0.0, np.pi, 100)
#    xc = radius[0]* np.outer(np.cos(u), np.sin(v))
#    yc = radius[1]*np.outer(np.sin(u), np.sin(v))
#    zc = radius[2]*np.outer(np.ones_like(u), np.cos(v))
#    
#    ##
#    a= np.kron(EigenVectors[:,0],xc)
#    b= np.kron(EigenVectors[:,1],yc)
#    c= np.kron(EigenVectors[:,2],zc)
#    
#    ##
#    data= a+b+c
#    n = data.shape[0]
#    
#    x = data[: , 0:n    ]     +Avg[0]
#    y = data[: , n:2*n  ]   +Avg[1]
#    z = data[: , 2*n:3*n] +Avg[2]
#    
#    ## plot the ellipsoïd
#    ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color='r', alpha=0.2)
##    ax.scatter3D(x,y,z,marker='o', c='r')
#
#    return
##=============================================================================
#def plot_2Dellipsoid(Avg, Var_Covar):
#    '''
#    plot a 2D-ellipsoïd in the current figure
#    used reference : http://stackoverflow.com/questions/12301071/multidimensional-confidence-intervals/12321306#12321306
#    
#    ax obtained thourgh : fig,ax = plt.subplos(), or ax=fig.gca() (and fig = plt.gcf())
#    
#    require  : from matplotlib.patches import Ellipse
#    Arguments:
#        Avg: 
#            NbClass x2 array with the average position of the ellipsoïd in a 2D space
#        
#        Var_CoVar:
#            NbClass x 2x2 matrix of Variance-Covariance
#        
#    Returns: 
#        Nothing
#    '''
#    ax=plt.gca()
#    for i in range(Avg.shape[0]):
#        
#        [EigenValues, EigenVectors] = np.linalg.eig(Var_Covar) ##U : eigenvalues (1x3) , V (3x3) : eigenvectors in columns
#        scale = 3.0
#        radius = scale * np.sqrt(EigenValues) # (1x2) 
#        theta  = np.degrees(np.arctan(*EigenVectors[:,0][::-1]) ) 
#        
#        ell  = Ellipse(     xy=Avg[i,:],
#                            width=radius[0], 
#                            height=radius[1], 
#                            angle=AngleClass[i],
#                            label = "class "+str(i)
#                            ) 
#        ax.add_artist(ellip)
#    
#    return
##=============================================================================
def plot_Spectral_Difference_Map(SpecDist)  :
    '''
        Plot the map of the spectral differences from the distance table defined as parameter
        difference with the BHSD : no approximation, all the samples are plotted
        
        old name : plot_BHSD(SpecDist)

        Parameters:
        ===========
            :param SpecDist: 2D table, 1 line per sample of shape difference and intensity difference
            :type SpecImg: float numpy array of NbSample x 2

        Returns:
        ========

    '''
    dist_shape  = SpecDist[:, 0]
    dist_energy = SpecDist[:, 1]
    
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    
    coloroption = np.array([[0,0.35,0.99,0.3]]) ## RGBA, A = alpha (transparence), 2D array with 1 line (see scatter documentation)

    plt.scatter( dist_shape[:], dist_energy[:], marker= 'o', c = coloroption)

    plt.xlabel('Shape difference from KLPD')
    plt.ylabel('Energy difference from KLPD')
    plt.title('BHSD')
    plt.show()
    
    return
##=============================================================================
def plot_BHSD(SpecDist, BHSD_title="", BHSDrange = None)  :
    '''
        Plot the BHSD on the current figure using the plt.hist2D() function from the distance table defined as parameter
        The title can be print after the function call if necessary
        
        Parameters:
        ===========
            :param SpecDist: 2D table, 1 line per sample of shape difference and intensity difference
            :type  float numpy array  nblg x 2 col
            :param BHSD_title : string to print as figure title
            :param BHSDrange : hist2D range :  [[xmin, xmax], [ymin, ymax]]
            :type  float numpy array of 2 x 2

        Returns:
        ========

    '''
    dist_shape  = SpecDist[:, 0]
    dist_energy = SpecDist[:, 1]
    
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()

    plt.hist2d(dist_shape[:], dist_energy[:], range = BHSDrange, bins = 150, density= True, norm=LogNorm())
        
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('# of spectral differences to the Reference', rotation=270)
   
    plt.xlabel('Shape Difference')
    plt.ylabel('Intensity Difference')
    plt.title(BHSD_title, fontsize=16)
    
    plt.show()
    
    return

##=============================================================================
def plot_3D_Spectral_Difference_Map(SpecDist1, SpecDist2, Fig_title, projected_axis = True)  :
    '''
        Plot the 3D Spectral Map (without using Histogram function) using 
        SpectDist1[:,0] for the x axis : shape difference to the ref1
        SpectDist2[:,0] for the y axis : shape difference to the ref2
        SpectDist1[:,1] for the z axis : intensity difference to the ref1
        
        if projected_axis = True : display the projection of the point cloud on the extremities of the 3D cube.
        
        Parameters:
        ===========
            :param SpecDist1: 2D table, 1 line per sample of shape difference and intensity difference
            :type  float numpy array  nblg x 2 col
            :param SpecDist2: 2D table, 1 line per sample of shape difference and intensity difference
            :type  float numpy array  nblg x 2 col
            :param BHSD_title : string to print as figure title
            :param projected_axis : if True the projection are displayed 
            :type  boolean (True or False)

        Returns:
        ========

    '''
    [MinX, MinZ] = SpecDist1.min(axis=0)
    [MaxY, poub] = SpecDist2.max(axis=0)
    
    coloroption = np.array([[0,0.35,0.99,0.3]]) ## scatter : RGBA, A = alpha (transparence), 2D array with 1 line (see scatter documentation)
    colorprojection =  (0.10,0.10,0.10, 0.1) ## plot, RGBA = tupple
    
    fig=plt.gcf()   ## take the current figure handler to create the axis and plot
    # plt.clf()     ## Clear the current figure
    ax = fig.add_subplot(111, projection='3d')
               
    ax.scatter(SpecDist1[:,0], SpecDist2[:,0], SpecDist1[:,1], c=coloroption, marker='o',alpha=0.15)
    if (projected_axis == True):
        ax.plot(SpecDist1[:,0], SpecDist1[:,1], '.', color = colorprojection, zdir='y', zs=MaxY)
        ax.plot(SpecDist2[:,0], SpecDist1[:,1], '.', color = colorprojection, zdir='x', zs=MinX)
        ax.plot(SpecDist1[:,0], SpecDist2[:,0], '.', color = colorprojection, zdir='z', zs=MinZ)


    ax.set_xlabel('Difference shape to the Min')
    ax.set_ylabel('Difference shape to the Max')
    ax.set_zlabel('Difference intensity to the Min (darker one)')
    plt.title(Fig_title, fontsize=16)
    
    plt.show()
    
    return

##=============================================================================


def Calc_XYZ2RGB_Matrix(Chromaticity_coordinate, Reference_White):
    '''
        Create a XYZ to RGB conversion Matrix given chromaticity coordinate and reference white of your RGB colorspace
        Based on Bruce Lindbloom equations : http://www.brucelindbloom.com/

        Parameters:
        ===========
            :param Chromaticity_coordinate: the chromaticity coordinate of your RGB space
            :type  Chromaticity_coordinate: float numpy array of shape (2, 3)

            | Xr, Yr |
            | Xg, Yg |
            | Xb, Yb |

            :param Reference_White: the reference white coordinate of your RGB space
            :type  Reference_White: 1D array of floats, length 3
        Returns:
        ========
            :return: Conversion matrix from XYZ to RGB space
            :rtype : float numpy array of shape (3, 3)
    '''
    XYZ_rgb = np.ones((3, 3))

    XYZ_rgb[:,0] = Chromaticity_coordinate[:,0] / Chromaticity_coordinate[:,1]
    XYZ_rgb[:,2] = (1 - Chromaticity_coordinate[:,0] - Chromaticity_coordinate[:,1]) / Chromaticity_coordinate[:,1]

    print(XYZ_rgb.T)

    S = np.linalg.inv(XYZ_rgb.T) @ Reference_White

    print(S)

    M = np.multiply(S, XYZ_rgb.T)

    return np.matrix(np.linalg.inv(M))


##=============================================================================

def XYZ_2_RGB_Space(ImgXYZ, RGB_space_matrix = None, Gamma=2.2):
    '''
        create an RGB image using the CIE-XYZ values 
        require to select an RGB space (sRGB, Adobe, CIE, ....)
        Based on Bruce Lindbloom equations : http://www.brucelindbloom.com/

        Parameters:
        ===========
            :param ImgXYZ: the Image in the primary space CIE XYZ (2°)
            :type  ImgXYZ: float numpy array of NbLG x NbCol x 2

            :param RGB_space_matrix : convertion matrix XYZ to RGB_space
            :type RGB_space_matrix  : 3x3 matrix of floats

            :param Gamma : Gamma value for the Non-linear visual adaptation (Gamma approx 2.2 for s-RGB), more important is Gamma more white is the image, less important more darky
            :type Gamma : numpy float
        Returns:
        ========
            :return: RgbImg with values in [0.0 , 1.0]
            :rtype : float numpy.array (nbLg, nbCol, 3)

    '''
    (lg, col, poub) = ImgXYZ.shape
    XYZ_list = ImgXYZ.reshape(lg*col,3)
        
    ImgRGB = XYZ_list * RGB_space_matrix.transpose()

    ## ramène les valeurs RGB dans la dynamique globale [0,1] : validée NR 15/10/2016
    mini = ImgRGB.min() ## attention : ne pas normaliser de façon marginal !!!!
    maxi = ImgRGB.max()
    ImgRGB= (ImgRGB - mini)/(maxi-mini)

    ## transforme la liste RGB en image lg x col x 3 canaux (R, G, B)
    ImgRGB = np.array(ImgRGB).reshape(lg,col,3)

    ## transformation Gamma pour ramener la dynamique du sRGB (Gamme =2.2 pour D65)
    ImgRGB= ImgRGB**(1/Gamma)

    return ImgRGB

##=============================================================================

def MonoSpec_2_CMF_XYZ(Waves, Spectrum):
    '''
        Return the chromaticity coordinates of a spectrum in the gamut
        Based on Bruce Lindbloom equations : http://www.brucelindbloom.com/

        Parameters:
        ===========
            :param Waves: the Spectrum sampling wavelength (nm)
            :type  Waves: float numpy array of NbCol (depend on sampling and wavelength interval)

            :param Spectrum: the Spectrum you want to caracterize
            :type  Spectrum: float numpy array of NbCol (depend on sampling and wavelength interval)

        Returns:
        ========
            :return: x,y the chromaticty coordinates in the gamut of your spectrum
            :rtype : a tuple of floats

    '''

    ## Identify the CIE and Ill wavelengthes to min and max of SpecImg. Adjust the range to it in the limit [360, 780] nm
    if (Waves[0] > 360.0):
        First_CIE_wave = ((Waves[0] - 360.0) / 5.0).astype(int) #The lowest wavelength of SpecImg > 360 nm
        First_SpecImg_wave = 0
#    elif (Waves[0] < 360.0):
#        First_CIE_wave = 0
#        First_SpecImg_wave = 0 #### To finish for a generic function  !!!!!
    else:
        First_CIE_wave = 0
    
    ## upper limits for the interpolation and then integrals
    if (Waves.max() > 780.0):
        tmp_wave, = np.where( Waves>780.0 )
        Last_SpecImg_wave = tmp_wave[0]
        Last_CIE_wave = 85              ## 84 correspond to 780.0
        
    elif (Waves.max() == 780.0):
        Last_SpecImg_wave = Waves.size   ##replace  Waves.size -1, the value correspond to the first index after the last one to keep
        Last_CIE_wave = 85 

    else:
        tmp_wave, = np.where( Waves == Waves.max() )
        Last_SpecImg_wave = tmp_wave[0]+1  ## replace tmp_wave[0], the value must be the nb of samples
        Last_CIE_wave = (np.ceil((Waves.max() -360.0)/5.0)).astype(int) ## must choose the upper integer in the list for the interpolation 

    # Load XYZ CMF
    CMF_X = np.load('data\\X_360_830_2deg.npy')
    CMF_Y = np.load('data\\Y_360_830_2deg.npy')
    CMF_Z = np.load('data\\Z_360_830_2deg.npy')

    Waves_CIE = np.arange(start = 360, stop = 835, step = 5)

    # Interpolate to match the sampling of the spectrum
    CMF_X_interp = np.interp(Waves[0:Last_SpecImg_wave],Waves_CIE,CMF_X)
    CMF_Y_interp = np.interp(Waves[0:Last_SpecImg_wave],Waves_CIE,CMF_Y)
    CMF_Z_interp = np.interp(Waves[0:Last_SpecImg_wave],Waves_CIE,CMF_Z)

    wavecount = Waves[0:Last_SpecImg_wave].size

    ## Transform the spectral image  in a table of lg x col spectra
    Spectral_list = np.matrix( Spectrum[0:Last_SpecImg_wave].reshape(1, wavecount) )

    ## Partie validée NR: 15/10/2016
    mat = np.matrix( [CMF_X_interp, CMF_Y_interp, CMF_Z_interp]  )
    XYZ = Spectral_list[0:Last_SpecImg_wave] * mat.transpose()
    XYZ = XYZ / np.sum(CMF_Y_interp)

    return XYZ

##=============================================================================
    
def MonoXYZ_2_RGB(XYZ, RGB_space ='sRGB', Gamma=2.2):
    '''
        create an RGB image using the CIE-XYZ values 
        require to select an RGB space (sRGB, Adobe, CIE, ....)
        Based on Bruce Lindbloom equations : http://www.brucelindbloom.com/

        Parameters:
        ===========
            :param XYZ: the XYZ values in the primary space CIE XYZ (2°)
            :type  XYZ: float numpy array of 1 x 3

            :param RGB_space : selected RGB space for the transform 
            :type RGB_space  : sRGB(D65), AdobeRGB(D65), other to define...

            :param Gamma : Gamma value for the Non-linear visual adaptation (Gamma approx 2.2 for s-RGB), more important is Gamma more white is the image, less important more darky
            :type Gamma : numpy float
        Returns:
        ========
            :return: RGB values with values in [0.0 , 1.0]
            :rtype : float numpy.array (1, 3)

    '''
    
    if (RGB_space =='AdobeRGB'): ## AdobeRGB - D65 , Bruce Lindbloom website
        Mm1 = np.matrix (   [ [2.0413690  , -0.5649464 , -0.3446944 ],
                              [-0.9692660 ,  1.8760108 ,  0.0415560 ],
                              [0.0134474  , -0.1183897 , 1.0154096  ] ]
                        )
    elif (RGB_space =='sRGB'): ## sRGB - D65 , Bruce Lindbloom website
        Mm1 = np.matrix (   [ [3.2404542 , -1.5371385 , -0.4985314 ],
                              [-0.9692660,  1.8760108 , 0.0415560],
                              [ 0.0556434, -0.2040259 , 1.0572252  ] ]
                        )
    else:
        print("the other matrix are not yet implemented !")
        
    RGB = XYZ * Mm1.transpose()

    ## ramène les valeurs RGB dans la dynamique globale [0,1] : validée NR 15/10/2016
    mini = RGB.min() ## attention : ne pas normaliser de faàon marginal !!!!
    maxi = RGB.max()
    RGB= (RGB - mini)/(maxi-mini)

    ## transformation Gamma pour ramener la dynamique du sRGB (Gamme =2.2 pour D65)
    RGB[0,0] = RGB[0,0] ** (1/Gamma)
    RGB[0,1] = RGB[0,1] ** (1/Gamma)
    RGB[0,2] = RGB[0,2] ** (1/Gamma)

    return RGB

##=============================================================================

def MonoXYZ_2_RGB_Space(XYZ, RGB_space_Matrix, Gamma=2.2):
    '''
        create an RGB image using the CIE-XYZ values 
        require to select an RGB space (sRGB, Adobe, CIE, ....)
        Based on Bruce Lindbloom equations : http://www.brucelindbloom.com/

        Parameters:
        ===========
            :param XYZ: the XYZ values in the primary space CIE XYZ (2°)
            :type  XYZ: float numpy array of 1 x 3

            :param RGB_space_matrix : convertion matrix XYZ to RGB_space
            :type RGB_space_matrix  : 3x3 matrix of floats

            :param Gamma : Gamma value for the Non-linear visual adaptation (Gamma approx 2.2 for s-RGB), more important is Gamma more white is the image, less important more darky
            :type Gamma : numpy float
        Returns:
        ========
            :return: RGB values with values in [0.0 , 1.0]
            :rtype : float numpy.array (1, 3)

    '''
        
    RGB = XYZ * RGB_space_Matrix.transpose()

    ## ramène les valeurs RGB dans la dynamique globale [0,1] : validée NR 15/10/2016
    mini = RGB.min() ## attention : ne pas normaliser de faàon marginal !!!!
    maxi = RGB.max()
    RGB= (RGB - mini)/(maxi-mini)

    ## transformation Gamma pour ramener la dynamique du sRGB (Gamme =2.2 pour D65)
    RGB[0,0] = RGB[0,0] ** (1/Gamma)
    RGB[0,1] = RGB[0,1] ** (1/Gamma)
    RGB[0,2] = RGB[0,2] ** (1/Gamma)

    return RGB