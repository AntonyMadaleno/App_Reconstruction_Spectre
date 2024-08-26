# -*- coding: utf-8 -*-

#================================================================================================================================================================================
# IMPORTS
#================================================================================================================================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import Tk, ttk, Label, Entry, Button, Frame, OptionMenu, StringVar, Listbox, END, Scrollbar, Checkbutton, IntVar, RIGHT, filedialog, colorchooser
from PIL import Image, ImageTk
import pandas as pd
import spectral as spy
import os
import cv2
import pywt
import pickle
import lzma

from libHSI.HSI_library import *
from libHSI.HSI_distances import *

from colour.plotting import plot_chromaticity_diagram_CIE1931

from reconstruction import *
from libHSI.HSI_reconstruction import *

#================================================================================================================================================================================
# App variable / access trough local functions only
#================================================================================================================================================================================

Bands               = np.array([])                                          # The bands loaded, a np array of shape (nblg, nbcol, nb_waves)
Band_ref            = np.array([])                                          # The reference band
Waves               = np.array([])                                          # the corresponding waves center in nm
Spectra             = np.array([])                                          # spectral response of system for each bands if available
Spectra_ref         = np.array([])                                          # reference spectrum
Resolution_Coef     = 1.0                                                   # Coefficient used to calculate pixel position on click

Theme_file          = "dark_theme.ini"                                      # relative filepath for the theme to load (color theme for the app)

#================================================================================================================================================================================
# Local Functions
#================================================================================================================================================================================

        #---------------#
        # I/O FUNCTIONS #
        #---------------#

#================================================================================================================================================================================

def loadEnvi_Compressed():
    '''
       load a spectral image and return an array of spectra

       Return:
           Dict         : the  Spy structure for hyperspectral img
           Img          : the spectral image as an array of spectra (nblg x nbcol x nb_wavelengths) of double real
           Wavelist     : the corresponding wavelength list
           Resolution   : the spectral resolution (for integration purpose) 

    '''

    # Step 1 Open a dialogue to ask for the file to be opened
    (f1, f2) = filedialog.askopenfilenames()

    if (f1.endswith(".hdr")):
        file_hdr = f1
        file_img = f2
    else:
        file_hdr = f2
        file_img = f1

    # Step 2 Open the header
    Dict = spy.io.envi.read_envi_header(file= file_hdr)                 # Open the header

    # Step 3 Load and uncompress the data
    Dwt = None
    with lzma.open(file_img, 'r') as f:
        Dwt = np.load(f)

    Slices = None
    with open(filedialog.askopenfilename(), 'rb') as f:
        Slices = pickle.load(f)

    Coeffs  = pywt.array_to_coeffs(Dwt, Slices, output_format='wavedec2')
    Img     = pywt.waverec2(Coeffs, 'db1')

    nbands  = int( Dict[ 'bands' ]   )
    nlines  = int( Dict[ 'lines' ]   )
    nrows   = int( Dict[ 'samples' ] ) 

    Img = Img[ :(nlines * nrows), :nbands]
    Img = Img.reshape( (nlines, nrows, nbands) )

    Wavelist    = np.array(Dict['wavelength']).astype(float)        # Get the wavelist
    Resolution  = Wavelist[1] - Wavelist[0]                         # Get the resolution of the sampling (only useful if sampling is homogenous)

    return (Dict, Img, Wavelist, Resolution)

#================================================================================================================================================================================

def loadEnvi():
    '''
       load a spectral image and return an array of spectra

       Return:
           Dict         : the  Spy structure for hyperspectral img
           Img          : the spectral image as an array of spectra (nblg x nbcol x nb_wavelengths) of double real
           Wavelist     : the corresponding wavelength list
           Resolution   : the spectral resolution (for integration purpose) 

    '''

    # Step 1 Open a dialogue to ask for the file to be opened
    (f1, f2) = filedialog.askopenfilenames()

    if (f1.endswith(".hdr")):
        file_hdr = f1
        file_img = f2
    else:
        file_hdr = f2
        file_img = f1

    # Step 2 Open the file using spy
    Dict = spy.io.envi.open(file= file_hdr, image= file_img)            # Open the data as spy structure
    Img = Dict.load()                                                   # Load the array of (nblg x nbcol x nb_waves) as numpy array of double
    Wavelist = np.array(Dict.bands.centers)                             # Get the wavelist
    Resolution = Wavelist[1] - Wavelist[0]                              # Get the resolution of the sampling (only useful if sampling is homogenous)

    return (Dict, Img, Wavelist, Resolution)

#================================================================================================================================================================================

def loadEnvi_with_Response():

    '''
       load a spectral image and return an array of spectra

       Return:
           Dict         : the  Spy structure for hyperspectral img
           Img          : the spectral image as an array of spectra (nblg x nbcol x nb_wavelengths) of double real
           Wavelist     : the corresponding wavelength list
           Resolution   : the spectral resolution (for integration purpose) 

    '''

    # Step 1 Open a dialogue to ask for the file to be opened
    (f1, f2) = filedialog.askopenfilenames()

    if (f1.endswith(".hdr")):
        file_hdr = f1
        file_img = f2
    else:
        file_hdr = f2
        file_img = f1

    # Step 2 Open the file using spy
    Dict = spy.io.envi.open(file= file_hdr, image= file_img)            # Open the data as spy structure
    Img = Dict.load()                                                   # Load the array of (nblg x nbcol x nb_waves) as numpy array of double
    Wavelist= np.array(Dict.bands.centers)                              # Get the wavelist
    Resolution = Wavelist[1] - Wavelist[0]                              # Get the resolution of the sampling (only useful if sampling is homogenous)

    response_file = filedialog.askopenfilenames()
    Responses = np.loadtxt(response_file)

    return (Dict, Img, Wavelist, Resolution)

    return

#================================================================================================================================================================================

def loadBand():
    '''
       load one band with its corresponding spd file and return an array of spectra

       Return:
           Band         : the band as an array of (nblg x nbcol)
           Spectra      : the spectral reponse of the system used to capture the band
           Wave         : the corresponding wavelength (center of Spectra)
    '''

    # Step 1 Open a dialogue to ask for the files to be opened
    file_img = filedialog.askopenfilename()
    file_spd = filedialog.askopenfilename()

    # Check data formats

    # Step 2 load the data
    Band    = cv2.imread(file_img, cv2.IMREAD_GRAYSCALE)        # Open the data as spy structure
    Spectra = np.loadtxt(file_spd).astype(np.double)            # Load sptral response from the system from .spd file

    # Calculate the center of the response signal
    half_sum = np.sum(Spectra[:,1]) / 2.0
    arg = 0; tmp = 0

    for i in range(0, Spectra.shape[0]):
        tmp += Spectra[i,1]

        if tmp >= half_sum:
            arg = i
            break

    Wave = Spectra[arg,0]

    return (Band, Spectra, Wave)

#================================================================================================================================================================================

def loadSpectrum():
    '''
       load one spectrum

       Return:
           Spectra      : the spectral reponse of the system used to capture the band [:,0] Waves [:,1] Spectra
    '''

    file_spd = filedialog.askopenfilename()

    # Check Data formats

    # Load spectral response from the system from .spd file
    return np.loadtxt(file_spd).astype(np.double)

#================================================================================================================================================================================

def saveEnvi(Bands, Waves):
    '''
        Save a ENVI format spectrale image

        Inputs:
            Bands       : Img Data as numpy floats array of (nblg, nbcol, nbwaves)
            Waves       : Wavelenghts in nanometers for each bands as numpy floats array of size (nbwaves)
    '''

    # Get the cube shape
    nbcol, nblg, nbwl   = Bands.shape

    # Create the Dict data
    Dict = {}
    Dict['lines']       = nblg
    Dict['samples']     = nbcol
    Dict['bands']       = nbwl

    if   (Bands.dtype == 'float64'):
        Dict['data_type']   = 5
    elif (Bands.dtype == 'float32'):
        Dict['data_type']   = 4
    elif (Bands.dtype == 'uint16'):
        Dict['data_type']   = 12
    elif (Bands.dtype == 'int16'):
        Dict['data_type']   = 2
    
    Dict['wavelength units']   = 'nm'
    Dict['wavelength'] = Waves

    Savepath = filedialog.asksaveasfilename()

    save_ENVI_Img(Bands, Dict, Savepath)

#================================================================================================================================================================================
# Tk Calls
#================================================================================================================================================================================

def Update_List():
    global Bands, Waves

    Band_list.delete(0, END)

    for Wave in Waves:
        Band_list.insert(END , f"Band {Wave}")

#================================================================================================================================================================================

def Load_ENVI():
    global Bands, Waves, Spectra
    __, Bands, Waves, __ = loadEnvi()
    Update_List()

#================================================================================================================================================================================

def Load_Band():
    global Bands, Waves, Spectra
    Band, Spectrum, Wave = loadBand()

    if (np.size(Bands) == 0):
        Bands = np.array([Band]).reshape((Band.shape[0], Band.shape[1], 1))
    else:
        Bands = np.append(Bands, Band.reshape((Band.shape[0], Band.shape[1], 1)), axis=2)

    if np.size(Waves) == 0:
        Spectra = np.array([Spectrum]).reshape((Spectrum.shape[0], Spectrum.shape[1], 1))
    else:
        Spectra = np.append(Spectra, Spectrum.reshape((Spectrum.shape[0], Spectrum.shape[1], 1)), axis=2)

    if np.size(Waves) == 0:
        Waves = np.array([Wave])
    else:
        Waves = np.append(Waves, Wave)

        
    Update_List()

#================================================================================================================================================================================

def Load_Ref_Band():
    global Band_ref, Spectra_ref
    Band_ref, Spectra_ref, __ = loadBand()

    Band_ref_255 = Band_ref.astype(float) * 255.0 / np.max(Band_ref.astype(float))

    img = Image.fromarray(Band_ref_255.astype(np.uint8), 'L')

    # Automatic resizing
    h, w = Band_ref.shape
    max_w = 720.0
    max_h = 480.0

    coef = min( max_w/float(w), max_h/float(h))

    nw, nh = round(coef * w), round(coef * h)

    img = img.resize( ( nw, nh ) )

    photo = ImageTk.PhotoImage(image=img)
    Ref_Image_display.config(image=photo)
    Ref_Image_display.image = photo

    root.update()

#================================================================================================================================================================================

def Load_Ref_Spectrum():
    global Spectra_ref
    Spectra_ref = loadSpectrum()

#================================================================================================================================================================================

def Save_ENVI():
    global Bands, Waves
    saveEnvi(Bands, Waves)

#================================================================================================================================================================================

def Save_RGB():
    RGB = Display_RGB().astype(np.uint8)
    cv2.imwrite(filename = filedialog.asksaveasfilename(defaultextension = 'png'), img = cv2.cvtColor(RGB, cv2.COLOR_RGB2BGR) )

#================================================================================================================================================================================

def Display_Band():
    global Bands, Waves
    
    # Get index of selected band in the list
    index = Band_list.curselection()[0]

    # Display the band in the display area of the app
    Band = Bands[:,:,index].reshape( (Bands.shape[0], Bands.shape[1]) )

    Band_255 = Band.astype(float) * 255.0 / np.max(Bands).astype(float)

    img = Image.fromarray(Band_255.astype(np.uint8), 'L')

    # Automatic resizing
    h, w = Band.shape
    max_w = 720.0
    max_h = 480.0

    coef = min( max_w/float(w), max_h/float(h))

    nw, nh = round(coef * w), round(coef * h)

    img = img.resize( ( nw, nh) )

    photo = ImageTk.PhotoImage(image=img)
    Image_display.config(image=photo)
    Image_display.image = photo

    root.update()
    
#================================================================================================================================================================================

def Display_RGB():

    global Bands

    if(Recal.get()):
        # Take first band as reference        
        Ref = Bands[:,:,0]

        # For each bands align
        for i in range (1, Bands.shape[2]):
            Bands[:,:,i] = Recalage(Ref, Bands[:,:,i])

    if (RGB_method.get() == "3 Bands"):
        return Display_RGB_3Bands()
    elif (RGB_method.get() == "Spectral_CMF"):
        return Display_RGB_CMF()
    elif (RGB_method.get() == "Spectral_Reconstruction Mono Channel"):
        return Display_Spectral_CMF()
    elif (RGB_method.get() == "Spectral_Reconstruction Multi Channel"):
        return Display_Spectral_CMF_3Channels()
    else:
        return Display_RGB_CMF()

#================================================================================================================================================================================
 
def Display_RGB_3Bands():
    global Bands, Waves
    
    # Get index of selected band in the list
    id_B, id_G, id_R = Band_list.curselection()[0], Band_list.curselection()[1], Band_list.curselection()[2]

    # Display the band in the display area of the app
    Band_R = Bands[:,:,id_R].reshape( (Bands.shape[0], Bands.shape[1]) )
    Band_G = Bands[:,:,id_G].reshape( (Bands.shape[0], Bands.shape[1]) )
    Band_B = Bands[:,:,id_B].reshape( (Bands.shape[0], Bands.shape[1]) )

    Band_R = Band_R * 255 / np.max(Bands[:,:,id_R])
    Band_G = Band_G * 255 / np.max(Bands[:,:,id_G])
    Band_B = Band_B * 255 / np.max(Bands[:,:,id_B])

    RGB = np.zeros((Band_R.shape[0], Band_R.shape[1], 3))

    RGB[:,:,0] = Band_R
    RGB[:,:,1] = Band_G
    RGB[:,:,2] = Band_B

    img = Image.fromarray(RGB.astype(np.uint8), 'RGB')

    # Automatic resizing
    h, w = Band_R.shape
    max_w = 720.0
    max_h = 480.0

    coef = min( max_w/float(w), max_h/float(h))

    nw, nh = round(coef * w), round(coef * h)

    img = img.resize( ( nw, nh ) )

    photo = ImageTk.PhotoImage(image=img)
    Image_display.config(image=photo)
    Image_display.image = photo

    root.update()

    return RGB
    
#================================================================================================================================================================================
 
def Display_RGB_CMF():
    global Bands, Waves, Spectra, Resolution_Coef, White_Correction
    
    # Get indices of selected bands in the list
    indices = Band_list.curselection()
    RGB = None

    # White correction (May lead to incorrect reconstructions) can be dactivated trough the GUI
    if (WC_method.get() == "Spectral White Correction"):

        WC_Bands = Bands / np.max(Bands, axis=(0,1))

        if len(indices) <= 0:
            RGB = multiSpec2CMF_RGB(WC_Bands, Waves, cmf='libHSI/data/#_360_830_2deg.npy', Gamma=float( Gamma_Entry.get() ))
        else:
            Selected_Bands = np.zeros( ( Bands[:,:,0].shape[0], Bands[:,:,0].shape[1], len(indices) ) )
            Selected_Waves = []
            for index in indices:  
                Selected_Bands[:,:,index] = Bands[:,:,index].reshape((Bands[:,:,index].shape[0], Bands[:,:,index].shape[1]))
                Selected_Waves.append(Waves[index])

            RGB = multiSpec2CMF_RGB(Selected_Bands, np.array(Selected_Waves), cmf='libHSI/data/#_360_830_2deg.npy', Gamma=float( Gamma_Entry.get() ))

        RGB = RGB * 255

    elif (WC_method.get() == "Spectral + RGB White Correction"):

        WC_Bands = Bands / np.max(Bands, axis=(0,1))

        if len(indices) <= 0:
            RGB = multiSpec2CMF_RGB(WC_Bands, Waves, cmf='libHSI/data/#_360_830_2deg.npy', Gamma=float( Gamma_Entry.get() ))
        else:
            Selected_Bands = np.zeros( ( Bands[:,:,0].shape[0], Bands[:,:,0].shape[1], len(indices) ) )
            Selected_Waves = []
            for index in indices:  
                Selected_Bands[:,:,index] = Bands[:,:,index].reshape((Bands[:,:,index].shape[0], Bands[:,:,index].shape[1]))
                Selected_Waves.append(Waves[index])

            RGB = multiSpec2CMF_RGB(Selected_Bands, np.array(Selected_Waves), cmf='libHSI/data/#_360_830_2deg.npy', Gamma=float( Gamma_Entry.get() ) )

        RGB[:,:,0] = ( RGB[:,:,0] - np.min(RGB[:,:,0]) ) * 255 / ( np.max(RGB[:,:,0]) - np.min(RGB[:,:,0]) )
        RGB[:,:,1] = ( RGB[:,:,1] - np.min(RGB[:,:,1]) ) * 255 / ( np.max(RGB[:,:,1]) - np.min(RGB[:,:,1]) )
        RGB[:,:,2] = ( RGB[:,:,2] - np.min(RGB[:,:,2]) ) * 255 / ( np.max(RGB[:,:,2]) - np.min(RGB[:,:,2]) )

    elif (WC_method.get() == "RGB White Correction"):

        if len(indices) <= 0:
            RGB = multiSpec2CMF_RGB(Bands, Waves, cmf='libHSI/data/#_360_830_2deg.npy', Gamma=float( Gamma_Entry.get() ))
        else:
            Selected_Bands = np.zeros( ( Bands[:,:,0].shape[0], Bands[:,:,0].shape[1], len(indices) ) )
            Selected_Waves = []
            for index in indices:  
                Selected_Bands[:,:,index] = Bands[:,:,index].reshape((Bands[:,:,index].shape[0], Bands[:,:,index].shape[1]))
                Selected_Waves.append(Waves[index])

            RGB = multiSpec2CMF_RGB(Selected_Bands, np.array(Selected_Waves), cmf='libHSI/data/#_360_830_2deg.npy', Gamma=float( Gamma_Entry.get() ) )

        RGB[:,:,0] = ( RGB[:,:,0] - np.min(RGB[:,:,0]) ) * 255 / ( np.max(RGB[:,:,0]) - np.min(RGB[:,:,0]) )
        RGB[:,:,1] = ( RGB[:,:,1] - np.min(RGB[:,:,1]) ) * 255 / ( np.max(RGB[:,:,1]) - np.min(RGB[:,:,1]) )
        RGB[:,:,2] = ( RGB[:,:,2] - np.min(RGB[:,:,2]) ) * 255 / ( np.max(RGB[:,:,2]) - np.min(RGB[:,:,2]) )

    else:

        if len(indices) <= 0:
            RGB = multiSpec2CMF_RGB(Bands, Waves, cmf='libHSI/data/#_360_830_2deg.npy', Gamma=float( Gamma_Entry.get() ))
        else:
            Selected_Bands = np.zeros( ( Bands[:,:,0].shape[0], Bands[:,:,0].shape[1], len(indices) ) )
            Selected_Waves = []
            for index in indices:  
                Selected_Bands[:,:,index] = Bands[:,:,index].reshape((Bands[:,:,index].shape[0], Bands[:,:,index].shape[1]))
                Selected_Waves.append(Waves[index])

            RGB = multiSpec2CMF_RGB(Selected_Bands, np.array(Selected_Waves), cmf='libHSI/data/#_360_830_2deg.npy', Gamma=float( Gamma_Entry.get() ))

        RGB = RGB * 255

    img = Image.fromarray(RGB.astype(np.uint8), 'RGB')

    # Automatic resizing
    h, w, c = RGB.shape
    max_w = 720.0
    max_h = 480.0

    Resolution_Coef = min( max_w/float(w), max_h/float(h))

    nw, nh = round(Resolution_Coef * w), round(Resolution_Coef * h)

    img = img.resize( (nw, nh) )

    photo = ImageTk.PhotoImage(image=img)
    Image_display.config(image=photo)
    Image_display.image = photo

    root.update()

    return RGB

#================================================================================================================================================================================
 
def Display_Spectral_CMF_3Channels():
    global Bands, Spectra, Resolution_Coef, White_Correction

    CMF_1 = np.array([loadSpectrum()])
    CMF_2 = np.array([loadSpectrum()])
    CMF_3 = np.array([loadSpectrum()])

    normalized_spectras, Energies   = normalize_wavelength_sampling(np.transpose(Spectra, (2,0,1)), low_bound = 380, high_bound = 780)

    normalized_CMF_1, Energy_1      = normalize_wavelength_sampling(CMF_1, low_bound = 380, high_bound = 780)
    Energy_1                        = Energy_1[0]

    normalized_CMF_2, Energy_2      = normalize_wavelength_sampling(CMF_2, low_bound = 380, high_bound = 780)
    Energy_2                        = Energy_2[0]

    normalized_CMF_3, Energy_3      = normalize_wavelength_sampling(CMF_3, low_bound = 380, high_bound = 780)
    Energy_3                        = Energy_3[0]

    if  ( Opti_method.get() == "Optimisation Stochastique" ):
        Coeffs_1 = HSI_Rec_Stochastique( CMF_1, Spectra, Criteria= Criteria.get(), low_norm= 380, high_norm= 780, definition= 1, steps = int( Step_Entry.get() ), nrand = int( Nrand_Entry.get() ) )
        Coeffs_2 = HSI_Rec_Stochastique( CMF_2, Spectra, Criteria= Criteria.get(), low_norm= 380, high_norm= 780, definition= 1, steps = int( Step_Entry.get() ), nrand = int( Nrand_Entry.get() ) )
        Coeffs_3 = HSI_Rec_Stochastique( CMF_3, Spectra, Criteria= Criteria.get(), low_norm= 380, high_norm= 780, definition= 1, steps = int( Step_Entry.get() ), nrand = int( Nrand_Entry.get() ) )

    elif( Opti_method.get() == "Optimisation par Recuit Simuler" ):
        Coeffs_1 = HSI_Rec_Recuit_Simuler( CMF_1, Spectra, Criteria= Criteria.get(), low_norm= 380, high_norm= 780, definition= 1, steps = int( Step_Entry.get() ) )
        Coeffs_2 = HSI_Rec_Recuit_Simuler( CMF_2, Spectra, Criteria= Criteria.get(), low_norm= 380, high_norm= 780, definition= 1, steps = int( Step_Entry.get() ) )
        Coeffs_3 = HSI_Rec_Recuit_Simuler( CMF_3, Spectra, Criteria= Criteria.get(), low_norm= 380, high_norm= 780, definition= 1, steps = int( Step_Entry.get() ) )

    elif( Opti_method.get() == "Optimisation multi_echelle" ):
        Coeffs_1 = HSI_Rec_Stochastique_Multi_Echelle( CMF_1, Spectra, Criteria= Criteria.get(), low_norm= 380, high_norm= 780, definition= 16, levels= 4, steps = int( Step_Entry.get() ) )
        Coeffs_2 = HSI_Rec_Stochastique_Multi_Echelle( CMF_2, Spectra, Criteria= Criteria.get(), low_norm= 380, high_norm= 780, definition= 16, levels= 4, steps = int( Step_Entry.get() ) )
        Coeffs_3 = HSI_Rec_Stochastique_Multi_Echelle( CMF_3, Spectra, Criteria= Criteria.get(), low_norm= 380, high_norm= 780, definition= 16, levels= 4, steps = int( Step_Entry.get() ) )

    # ---

    New_Bands = np.zeros(( Bands.shape[0], Bands.shape[1], 3 ))

    for k in range (Bands.shape[2]):
        New_Bands[:,:,0] += Bands[:,:,k] * Coeffs_1[k]
        New_Bands[:,:,1] += Bands[:,:,k] * Coeffs_2[k]
        New_Bands[:,:,2] += Bands[:,:,k] * Coeffs_3[k]

    

    if (WC_method.get() == "RGB White Correction"):
        New_Bands[:,:,0] /= np.max( New_Bands[:,:,0] )
        New_Bands[:,:,1] /= np.max( New_Bands[:,:,1] )
        New_Bands[:,:,2] /= np.max( New_Bands[:,:,2] )
    else:
        New_Bands /= np.max(New_Bands)

    New_Bands = New_Bands ** (1 / float( Gamma_Entry.get() ) )
    New_Bands *= 255

    img = Image.fromarray(New_Bands.astype(np.uint8), 'RGB')

    # Automatic resizing
    h, w, c = New_Bands.shape
    max_w = 720.0
    max_h = 480.0

    Resolution_Coef = min( max_w/float(w), max_h/float(h))

    nw, nh = round(Resolution_Coef * w), round(Resolution_Coef * h)

    img = img.resize( (nw, nh) )

    photo = ImageTk.PhotoImage(image=img)
    Image_display.config(image=photo)
    Image_display.image = photo

    root.update()

    def measure_LAP(img, ksize = 3, ddepth = cv2.CV_64F):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = cv2.Laplacian(img, ddepth, ksize = ksize)
        result = cv2.convertScaleAbs(result)

        w, h = img.shape

        return np.sum(result)

    return New_Bands

#================================================================================================================================================================================
 
def Display_Spectral_CMF():
    global Bands, Spectra, Resolution_Coef, White_Correction

    CMF = np.array([loadSpectrum()])

    normalized_spectras, Energies   = normalize_wavelength_sampling(np.transpose(Spectra, (2,0,1)), low_bound = 380, high_bound = 780)
    normalized_CMF, Energy          = normalize_wavelength_sampling(CMF, low_bound = 380, high_bound = 780)
    normalized_CMF                  = normalized_CMF[0]
    Energy                          = Energy[0]

    if  ( Opti_method.get() == "Optimisation Stochastique" ):
        Coeffs = HSI_Rec_Stochastique(CMF, Spectra, Criteria= Criteria.get(), low_norm= 380, high_norm= 780, definition= 1, steps = int( Step_Entry.get() ), nrand = int( Nrand_Entry.get() ), debug= True )

    elif( Opti_method.get() == "Optimisation par Recuit Simuler" ):
        Coeffs = HSI_Rec_Recuit_Simuler(CMF, Spectra, Criteria= Criteria.get(), low_norm= 380, high_norm= 780, definition= 1, steps = int( Step_Entry.get() ), debug= True )

    elif( Opti_method.get() == "Optimisation multi_echelle" ):
        Coeffs = HSI_Rec_Stochastique_Multi_Echelle(CMF, Spectra, Criteria= Criteria.get(), low_norm= 380, high_norm= 780, definition= 16, levels= 4, steps = int( Step_Entry.get() ), debug= True )

    # ---

    New_Band = np.zeros(( Bands.shape[0], Bands.shape[1] ))

    for k in range (Bands.shape[2]):
        New_Band += Bands[:,:,k] * Coeffs[k]

    New_Band *= 255 / np.max( New_Band )

    img = Image.fromarray(New_Band.astype(np.uint8), 'L')

    # Automatic resizing
    h, w = New_Band.shape
    max_w = 720.0
    max_h = 480.0

    Resolution_Coef = min( max_w/float(w), max_h/float(h))

    nw, nh = round(Resolution_Coef * w), round(Resolution_Coef * h)

    img = img.resize( (nw, nh) )

    photo = ImageTk.PhotoImage(image=img)
    Image_display.config(image=photo)
    Image_display.image = photo

    root.update()

    return New_Band

#================================================================================================================================================================================

def Display_Pixel_Spectrum(event):
    global Bands, Waves, Ax_Chroma, Ax_Spectrum, Fig_Chroma, Fig_Spectrum, Canvas_Chroma, Canvas_Spectrum

    x = int(event.y / Resolution_Coef)
    y = int(event.x / Resolution_Coef)

    Spec = Bands[x,y,:].flatten()

    RGB = None

    if (WC_method.get() == "Spectral White Correction"):
        Spec = Spec / np.max(Bands, axis=(0,1))
        RGB = monoSpec2CMF_RGB(Spec, Waves, Gamma=float( Gamma_Entry.get() ), cmf='libHSI/data/#_360_830_2deg.npy')
    else:
        Spec = Spec / np.max(Bands)
        RGB = monoSpec2CMF_RGB(Spec, Waves, Gamma=float( Gamma_Entry.get() ), cmf='libHSI/data/#_360_830_2deg.npy')

    MsRGB = np.matrix([
        [0.412,     0.358,      0.180],
        [0.213,     0.715,      0.072],
        [0.019,     0.119,      0.950]
    ])

    XYZ = MsRGB @ RGB.T
    RGB = RGB * 255 / np.max(RGB)

    cx, cy = XYZ[0,0] / np.sum(XYZ), XYZ[1,0] / np.sum(XYZ)

    RGB = (RGB).astype(np.uint8)
    hex_color = '#%02x%02x%02x' % (RGB[0,0], RGB[0,1], RGB[0,2])

    Ax_Chroma.clear()
    Fig_Chroma, Ax_Chroma = plot_chromaticity_diagram_CIE1931(show= False, spectral_locus_opacity=0.5, diagram_opacity=0.5)
    Canvas_Chroma.figure = Fig_Chroma

    Ax_Chroma.set_ylabel('y')
    Ax_Chroma.set_xlabel('x')
    Ax_Chroma.set_title('CIE 1931 Chromaticity Diagram', color=Theme['color_3'])

    Ax_Chroma.plot(cx, cy, 'o', color=hex_color, label="Estimation chromaticity", alpha = 0.75) # Chroma

    # Style
    Fig_Chroma.set_facecolor(Theme['color_0'])
    Fig_Chroma.get_axes()[0].set_facecolor(Theme['color_1'])
    Fig_Chroma.get_axes()[0].spines['bottom'].set_color(Theme['color_3'])
    Fig_Chroma.get_axes()[0].spines['top'].set_color(Theme['color_3'])
    Fig_Chroma.get_axes()[0].spines['right'].set_color(Theme['color_3'])
    Fig_Chroma.get_axes()[0].spines['left'].set_color(Theme['color_3'])
    Fig_Chroma.get_axes()[0].xaxis.label.set_color(Theme['color_3'])
    Fig_Chroma.get_axes()[0].yaxis.label.set_color(Theme['color_3'])
    Fig_Chroma.get_axes()[0].tick_params(axis='x', colors=Theme['color_3'])
    Fig_Chroma.get_axes()[0].tick_params(axis='y', colors=Theme['color_3'])

    Ax_Spectrum.clear()

    Ax_Spectrum.set_xlabel('wavelength (nm)', color=Theme['color_4'])
    Ax_Spectrum.set_ylabel('Y', color=Theme['color_4'])
    Ax_Spectrum.set_title('Intensity per wavelength', color=Theme['color_4'])
    Ax_Spectrum.grid(True)

    Fig_Spectrum.get_axes()[0].set_ylim((0.0, 1.0))

    Ax_Spectrum.plot(Waves, Spec, alpha = 0.5, lw = 4, label = f"pixel({x},{y})", color = hex_color)

    Canvas_Spectrum.draw()
    Canvas_Chroma.draw()
    return

#================================================================================================================================================================================
# TK
#================================================================================================================================================================================

#================================================================================================================================================================================

Theme = {}

with open(Theme_file, 'r') as file:

    for line in file:
        key, value  = line.strip().split(',')
        key         = key.strip()
        value       = value.strip()

        if key == 'font_size':
            value = int(value)
        
        Theme[key] = value

default_font = (Theme['font_family'], Theme['font_size'])

#================================================================================================================================================================================

# Create a Tkinter window
root = Tk()
root.configure(background= Theme['color_0'])

#================================================================================================================================================================================
# Left Pannel for user controls 
#   - Load Envi | Bands | Reference Spectrum | Referennce Band
#   - Select Band
#   - Select RGB reconstruction method
#   - Gamma input
#   - RGB reconstruction
#================================================================================================================================================================================
left_pannel = Frame(root, bg = Theme['color_1'])
left_pannel.grid(column= 0, row= 0, sticky= tk.EW, padx = 0, pady = 0)

### LOAD ENVI ###
LoadEnvi_button = Button(left_pannel, text="Load ENVI", command=Load_ENVI, font=default_font, relief = "flat", bg = Theme['color_4'], activebackground = Theme['color_3'])
LoadEnvi_button.grid(row= 0, column= 0, columnspan= 2, sticky= tk.EW, padx = 5, pady = 5)

### LOAD BAND ###
LoadEnvi_button = Button(left_pannel, text="Load Band", command=Load_Band, font=default_font, relief = "flat", bg = Theme['color_4'], activebackground = Theme['color_3'])
LoadEnvi_button.grid(row= 1, column= 0, columnspan= 2, sticky= tk.EW, padx = 5, pady = 5)

### LOAD REFERENCE IMAGE ###
LoadEnvi_button = Button(left_pannel, text="Load Reference Band", command=Load_Ref_Band, font=default_font, relief = "flat", bg = Theme['color_4'], activebackground = Theme['color_3'])
LoadEnvi_button.grid(row= 2, column= 0, columnspan= 2, sticky= tk.EW, padx = 5, pady = 5)

### LOAD REFERENCE SPECTRUM ###
LoadEnvi_button = Button(left_pannel, text="Load Reference Spectrum", command=Load_Ref_Spectrum, font=default_font, relief = "flat", bg = Theme['color_4'], activebackground = Theme['color_3'])
LoadEnvi_button.grid(row= 3, column= 0, columnspan= 2, sticky= tk.EW, padx = 5, pady = 5)

### SAVE ENVI ###
LoadEnvi_button = Button(left_pannel, text="Save ENVI", command=Save_ENVI, font=default_font, relief = "flat", bg = Theme['color_4'], activebackground = Theme['color_3'])
LoadEnvi_button.grid(row= 4, column= 0, columnspan= 2, sticky= tk.EW, padx = 5, pady = 5)

### SAVE RGB / BAND ###
LoadEnvi_button = Button(left_pannel, text="Save RGB", command=Save_RGB, font=default_font, relief = "flat", bg = Theme['color_4'], activebackground = Theme['color_3'])
LoadEnvi_button.grid(row= 5, column= 0, columnspan= 2, sticky= tk.EW, padx = 5, pady = 5)

### DISPLAY BAND LIST ###
Band_list = Listbox(left_pannel, selectmode="multiple", font=default_font, foreground= Theme['color_3'], bg = Theme['color_2'], width = 50)
Band_list.grid(row= 6, rowspan = 6, column= 0, columnspan= 2, sticky= tk.EW, padx = 5, pady = 5)

scrollbar = Scrollbar(left_pannel, orient="vertical")
scrollbar.config(command = Band_list.yview)
scrollbar.grid(row= 6, rowspan= 6, column= 2, padx = 5, pady = 5)

Band_list.config(yscrollcommand=scrollbar.set)

### SHOW SELECTED BAND ###
Select_Band_button = Button(left_pannel, text="Display Selected Band", command=Display_Band, font=default_font, relief = "flat", bg = Theme['color_4'], activebackground = Theme['color_3'])
Select_Band_button.grid(row= 12, column= 0,  columnspan= 2, sticky= tk.EW, padx = 5, pady = 5)

### SHOW RGB Reconstruction (Result depend on selected reconstruction method) ###
Select_Band_button = Button(left_pannel, text="Reconstruction", command=Display_RGB, font=default_font, relief = "flat", bg = Theme['color_4'], activebackground = Theme['color_3'])
Select_Band_button.grid(row= 13, column= 0, columnspan= 1, sticky= tk.EW, padx = 5, pady = 5)

# Activation of Recalage
Recal = tk.BooleanVar()

Toggle_Recalage = ttk.Checkbutton(left_pannel, text= "Registration", variable= Recal)
Toggle_Recalage.grid(row= 13, column= 1, columnspan= 1, sticky= tk.EW, padx = 5, pady = 5)

### RGB Reconstruction Method Selection ###
RGB_methods_names = ["3 Bands", "Spectral_CMF", "Spectral_Reconstruction Mono Channel", "Spectral_Reconstruction Multi Channel"]

RGB_method = StringVar()
RGB_method.set(RGB_methods_names[0])

RGB_method_menu = OptionMenu(left_pannel, RGB_method, *RGB_methods_names)
RGB_method_menu.grid(row = 14, column= 0, columnspan= 2, sticky= tk.EW, padx= 5, pady= 5)

### Optimisation method ###
Opti_methods_names = ["Non Négative Least Square", "Optimisation Stochastique", "Optimisation par Recuit Simuler", "Optimisation multi_echelle"]

Opti_method = StringVar()
Opti_method.set(Opti_methods_names[0])

Opti_method_menu = OptionMenu(left_pannel, Opti_method, *Opti_methods_names)
Opti_method_menu.grid(row = 15, column= 0, columnspan= 2, sticky= tk.EW, padx= 5, pady= 5)

### White Correction Method (Default RGB) ###
WC_methods_names = ["No White Correction", "RGB White Correction", "Spectral White Correction", "Spectral + RGB White Correction"]

WC_method = StringVar()
WC_method.set(WC_methods_names[1])

WC_method_menu = OptionMenu(left_pannel, WC_method, *WC_methods_names)
WC_method_menu.grid(row = 16, column= 0, columnspan= 2, sticky= tk.EW, padx= 5, pady= 5)

### Gamma ###
Label(left_pannel, text="Gamma :", font=default_font, foreground= Theme['color_3'], bg = Theme['color_1']).grid(row=17, column=0, sticky='w', padx=5, pady=5)
Gamma_Entry = Entry(left_pannel, font=default_font, foreground= Theme['color_3'], bg = Theme['color_0'])
Gamma_Entry.insert(1, "1.0")
Gamma_Entry.grid(row = 17, column= 1, sticky= tk.EW, padx= 5, pady= 5)

### Step Count ###
Label(left_pannel, text="Step Count :", font=default_font, foreground= Theme['color_3'], bg = Theme['color_1']).grid(row=18, column=0, sticky='w', padx=5, pady=5)
Step_Entry = Entry(left_pannel, font=default_font, foreground= Theme['color_3'], bg = Theme['color_0'])
Step_Entry.insert(100, "100")
Step_Entry.grid(row = 18, column= 1, sticky= tk.EW, padx= 5, pady= 5)

### N Random ###
Label(left_pannel, text="Sample Count per Step :", font=default_font, foreground= Theme['color_3'], bg = Theme['color_1']).grid(row=19, column=0, sticky='w', padx=5, pady=5)
Nrand_Entry = Entry(left_pannel, font=default_font, foreground= Theme['color_3'], bg = Theme['color_0'])
Nrand_Entry.insert(10000, "10000")
Nrand_Entry.grid(row = 19, column= 1, sticky= tk.EW, padx= 5, pady= 5)

### Reconstruction Criteria ###
Criterias = ["SAM", "SGA", "SID", "SCA", "Minkowski_L1", "Minkowski_L2"]

Criteria = StringVar()
Criteria.set(Criterias[0])

Criteria_menu = OptionMenu(left_pannel, Criteria, *Criterias)
Criteria_menu.grid(row = 20, column= 0, columnspan= 2, sticky= tk.EW, padx= 5, pady= 5)

#================================================================================================================================================================================
# Center Pannel for display band / image ...
#================================================================================================================================================================================
center_pannel = Frame(root, bg = Theme['color_2'])
center_pannel.grid(column= 1, row= 0, columnspan= 3, sticky= tk.EW, padx = 5, pady = 5)

### DISPLAY IMAGE ###
Image_display = Label(center_pannel, bg=Theme['color_3'])
Image_display.grid(column= 0, row= 0, rowspan= 3, sticky= tk.EW, padx = 5, pady = 5)
Image_display.bind("<Button-1>", Display_Pixel_Spectrum)

### DISPLAY Reference IMAGE ###
Ref_Image_display = Label(center_pannel, bg=Theme['color_3'])
Ref_Image_display.grid(column= 0, row= 4, rowspan= 3, sticky= tk.EW, padx = 5, pady = 5)

#================================================================================================================================================================================
# Center Pannel for display band / image ...
#================================================================================================================================================================================
right_pannel = Frame(root, bg = Theme['color_2'])
right_pannel.grid(column= 4, row= 0, columnspan= 1, sticky= tk.EW, padx = 5, pady = 5)

### DISPLAY SPECTRUM ###

Fig_Spectrum, Ax_Spectrum = plt.subplots()
Canvas_Spectrum = FigureCanvasTkAgg(Fig_Spectrum, master=right_pannel)
Canvas_Spectrum.get_tk_widget().grid(row = 0, column = 0, padx = 5, pady = 5)

# Style
Fig_Spectrum.set_facecolor(Theme['color_0'])
Fig_Spectrum.get_axes()[0].set_facecolor(Theme['color_1'])
Fig_Spectrum.get_axes()[0].spines['bottom'].set_color(Theme['color_3'])
Fig_Spectrum.get_axes()[0].spines['top'].set_color(Theme['color_3'])
Fig_Spectrum.get_axes()[0].spines['right'].set_color(Theme['color_3'])
Fig_Spectrum.get_axes()[0].spines['left'].set_color(Theme['color_3'])
Fig_Spectrum.get_axes()[0].xaxis.label.set_color(Theme['color_3'])
Fig_Spectrum.get_axes()[0].yaxis.label.set_color(Theme['color_3'])
Fig_Spectrum.get_axes()[0].tick_params(axis='x', colors=Theme['color_3'])
Fig_Spectrum.get_axes()[0].tick_params(axis='y', colors=Theme['color_3'])

### DISPLAY Chromaticity ###

Fig_Chroma, Ax_Chroma = plot_chromaticity_diagram_CIE1931(show=False, spectral_locus_opacity=0.5, diagram_opacity=0.5)
Canvas_Chroma = FigureCanvasTkAgg(Fig_Chroma, master=right_pannel)
Canvas_Chroma.get_tk_widget().grid(row = 1, column = 0, padx = 5, pady = 5)

# Style
Fig_Chroma.set_facecolor(Theme['color_0'])
Fig_Chroma.get_axes()[0].set_facecolor(Theme['color_1'])
Fig_Chroma.get_axes()[0].spines['bottom'].set_color(Theme['color_3'])
Fig_Chroma.get_axes()[0].spines['top'].set_color(Theme['color_3'])
Fig_Chroma.get_axes()[0].spines['right'].set_color(Theme['color_3'])
Fig_Chroma.get_axes()[0].spines['left'].set_color(Theme['color_3'])
Fig_Chroma.get_axes()[0].xaxis.label.set_color(Theme['color_3'])
Fig_Chroma.get_axes()[0].yaxis.label.set_color(Theme['color_3'])
Fig_Chroma.get_axes()[0].tick_params(axis='x', colors=Theme['color_3'])
Fig_Chroma.get_axes()[0].tick_params(axis='y', colors=Theme['color_3'])

#================================================================================================================================================================================

# Run the Tkinter event loop
root.mainloop()
