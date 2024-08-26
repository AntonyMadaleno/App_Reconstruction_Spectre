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
import pandas as pd

from libHSI.HSI_library import *
from libHSI.HSI_distances import *

from colour.plotting import plot_chromaticity_diagram_CIE1931

from reconstruction import *
from libHSI.HSI_reconstruction import *

#================================================================================================================================================================================
# App variable / access trough local functions only
#================================================================================================================================================================================

Waves               = np.array([])                                          # the corresponding waves center in nm
Spectra             = np.array([])                                          # spectral response of system for each bands if available
Spectra_Ref         = np.array([])                                          # reference spectrum

Theme_file          = "dark_theme.ini"                                      # relative filepath for the theme to load (color theme for the app)

#================================================================================================================================================================================
# Local Functions
#================================================================================================================================================================================

        #---------------#
        # I/O FUNCTIONS #
        #---------------#

#================================================================================================================================================================================

def loadSpectra():
    '''
       load one band with its corresponding spd file and return an array of spectra

       Return:
           Spectra      : the spectral reponse of the system used to capture the band
           Wave         : the corresponding wavelength (center of Spectra)
    '''

    # Step 1 Open a dialogue to ask for the files to be opened
    file_spd = filedialog.askopenfilename(title="load spectrum")

    # Check data formats

    # Step 2 load the data
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

    return (Spectra, Wave)

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
# Tk Calls
#================================================================================================================================================================================

def Update_List():
    global Waves

    Band_list.delete(0, END)

    for Wave in Waves:
        Band_list.insert(END , f"Band {Wave}")

#================================================================================================================================================================================

def Load_Spectra():
    global Waves, Spectra
    Spectrum, Wave = loadSpectra()

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

def Display_Spectra():
    global Waves, Spectra

    Ax_Spectrum.clear()
    for i in range( Spectra.shape[2] ):
        x = Spectra[:,0,i]
        y = Spectra[:,1,i]
        name = Waves[i]
        Ax_Spectrum.plot(x, y, alpha = 0.5, lw = 5, label=name)

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

    plt.tight_layout()
    Canvas_Spectrum.draw()

    root.update()

#================================================================================================================================================================================

def Load_Spectrums():
    global Waves, Spectra
    Spectrum, Wave = loadSpectra()

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

def Load_Ref_Spectrum():
    global Spectra_Ref
    Spectra_Ref = loadSpectrum()
    
#================================================================================================================================================================================

def Calculate_Coeffs():

    K, info =  Calculate_Optimisation()

    df = pd.DataFrame(info)
    df.to_csv(filedialog.asksaveasfilename(), index=False)

    # Générer des graphes avec matplotlib
    for key in info.keys():
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df[key], label=key)

        plt.xlabel('Steps')
        plt.ylabel('Scores')
        plt.title('Scores Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(filedialog.asksaveasfilename(title=f"save {key} score"))

        Ax_Result.clear()
        plt.close()

    normalized_spectras, E_Samples   = normalize_wavelength_sampling(np.transpose(Spectra, (2,0,1)), low_bound = 380, high_bound = 760)
    normalized_CMF, Energy          = normalize_wavelength_sampling(np.array([Spectra_Ref]), low_bound = 380, high_bound = 760)
    normalized_CMF                  = normalized_CMF[0]
    Energy                          = Energy[0]

    samples = []
    tmp = np.copy(K)

    for i in range(len(normalized_spectras)):
        samples.append(np.array(normalized_spectras[i][1] * E_Samples[i]))
        K[i] = K[i] * Energy / ( np.sum(tmp * E_Samples) )

    print( (K * 100).astype(np.uint16) )

    samples = np.array(samples)

    y = np.dot(K, samples)

    Ax_Result.plot(normalized_CMF[0], y, alpha = 0.5, lw = 4, label="Reconstruction result", color="#FF0000")
    Ax_Result.plot(normalized_CMF[0], normalized_CMF[1] * Energy, alpha = 0.5, lw = 4, label="Target", color="#00FF66")

    # Style
    Fig_Result.set_facecolor(Theme['color_0'])
    Fig_Result.get_axes()[0].set_facecolor(Theme['color_1'])
    Fig_Result.get_axes()[0].spines['bottom'].set_color(Theme['color_3'])
    Fig_Result.get_axes()[0].spines['top'].set_color(Theme['color_3'])
    Fig_Result.get_axes()[0].spines['right'].set_color(Theme['color_3'])
    Fig_Result.get_axes()[0].spines['left'].set_color(Theme['color_3'])
    Fig_Result.get_axes()[0].xaxis.label.set_color(Theme['color_3'])
    Fig_Result.get_axes()[0].yaxis.label.set_color(Theme['color_3'])
    Fig_Result.get_axes()[0].tick_params(axis='x', colors=Theme['color_3'])
    Fig_Result.get_axes()[0].tick_params(axis='y', colors=Theme['color_3'])

    plt.legend()

    plt.tight_layout()
    Canvas_Result.draw()



#================================================================================================================================================================================
 
def Calculate_Optimisation():
    global Spectra

    CMF = np.array([Spectra_Ref])

    Coeffs = None

    if  ( Opti_method.get() == "Optimisation Stochastique" ):
        Coeffs = HSI_Rec_Stochastique(CMF, Spectra, Criteria= Criteria.get(), low_norm= 380, high_norm= 760, definition= 1, steps = int( Step_Entry.get() ), nrand = int( Nrand_Entry.get() ), debug= True )

    elif( Opti_method.get() == "Optimisation par Recuit Simuler" ):
        Coeffs = HSI_Rec_Recuit_Simuler(CMF, Spectra, Criteria= Criteria.get(), low_norm= 380, high_norm= 760, definition= 1, steps = int( Step_Entry.get() ), debug= True )

    elif( Opti_method.get() == "Optimisation multi_echelle" ):
        Coeffs = HSI_Rec_Stochastique_Multi_Echelle(CMF, Spectra, Criteria= Criteria.get(), low_norm= 380, high_norm= 760, definition= 16, levels= 4, steps = int( Step_Entry.get() ), debug= True )

    root.update()

    return Coeffs

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

### LOAD BAND ###
LoadEnvi_button = Button(left_pannel, text="Load Spectra", command=Load_Spectra, font=default_font, relief = "flat", bg = Theme['color_4'], activebackground = Theme['color_3'])
LoadEnvi_button.grid(row= 0, column= 0, columnspan= 2, sticky= tk.EW, padx = 5, pady = 5)

### LOAD REFERENCE SPECTRUM ###
LoadEnvi_button = Button(left_pannel, text="Load Reference Spectra", command=Load_Ref_Spectrum, font=default_font, relief = "flat", bg = Theme['color_4'], activebackground = Theme['color_3'])
LoadEnvi_button.grid(row= 1, column= 0, columnspan= 2, sticky= tk.EW, padx = 5, pady = 5)

### DISPLAY BAND LIST ###
Band_list = Listbox(left_pannel, selectmode="multiple", font=default_font, foreground= Theme['color_3'], bg = Theme['color_2'], width = 50)
Band_list.grid(row= 2, rowspan = 6, column= 0, columnspan= 2, sticky= tk.EW, padx = 5, pady = 5)

scrollbar = Scrollbar(left_pannel, orient="vertical")
scrollbar.config(command = Band_list.yview)
scrollbar.grid(row= 2, rowspan= 6, column= 2, padx = 5, pady = 5)

Band_list.config(yscrollcommand=scrollbar.set)

### SHOW SELECTED SPECTRA ###
Select_Band_button = Button(left_pannel, text="Display Selected Band", command=Display_Spectra, font=default_font, relief = "flat", bg = Theme['color_4'], activebackground = Theme['color_3'])
Select_Band_button.grid(row= 8, column= 0,  columnspan= 2, sticky= tk.EW, padx = 5, pady = 5)

### SHOW SPECTRA Reconstruction (Result depend on selected reconstruction method) ###
Select_Band_button = Button(left_pannel, text="Reconstruction", command=Calculate_Coeffs, font=default_font, relief = "flat", bg = Theme['color_4'], activebackground = Theme['color_3'])
Select_Band_button.grid(row= 9, column= 0, columnspan= 1, sticky= tk.EW, padx = 5, pady = 5)

### Optimisation method ###
Opti_methods_names = ["Non Négative Least Square", "Optimisation Stochastique", "Optimisation par Recuit Simuler", "Optimisation multi_echelle"]

Opti_method = StringVar()
Opti_method.set(Opti_methods_names[0])

Opti_method_menu = OptionMenu(left_pannel, Opti_method, *Opti_methods_names)
Opti_method_menu.grid(row = 10, column= 0, columnspan= 2, sticky= tk.EW, padx= 5, pady= 5)

### Step Count ###
Label(left_pannel, text="Step Count :", font=default_font, foreground= Theme['color_3'], bg = Theme['color_1']).grid(row=11, column=0, sticky='w', padx=5, pady=5)
Step_Entry = Entry(left_pannel, font=default_font, foreground= Theme['color_3'], bg = Theme['color_0'])
Step_Entry.insert(100, "100")
Step_Entry.grid(row = 11, column= 1, sticky= tk.EW, padx= 5, pady= 5)

### N Random ###
Label(left_pannel, text="Sample Count per Step :", font=default_font, foreground= Theme['color_3'], bg = Theme['color_1']).grid(row=12, column=0, sticky='w', padx=5, pady=5)
Nrand_Entry = Entry(left_pannel, font=default_font, foreground= Theme['color_3'], bg = Theme['color_0'])
Nrand_Entry.insert(10000, "10000")
Nrand_Entry.grid(row = 12, column= 1, sticky= tk.EW, padx= 5, pady= 5)

### Reconstruction Criteria ###
Criterias = ["SAM", "SGA", "SID", "SCA", "Minkowski_L1", "Minkowski_L2"]

Criteria = StringVar()
Criteria.set(Criterias[0])

Criteria_menu = OptionMenu(left_pannel, Criteria, *Criterias)
Criteria_menu.grid(row = 13, column= 0, columnspan= 2, sticky= tk.EW, padx= 5, pady= 5)

#================================================================================================================================================================================
# Center Pannel for display band / image ...
#================================================================================================================================================================================
center_pannel = Frame(root, bg = Theme['color_2'])
center_pannel.grid(column= 1, row= 0, columnspan= 3, sticky= tk.EW, padx = 5, pady = 5)

### DISPLAY SPECTRUM ###

Fig_Spectrum, Ax_Spectrum = plt.subplots()
Canvas_Spectrum = FigureCanvasTkAgg(Fig_Spectrum, master=center_pannel)
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

Fig_Result, Ax_Result = plt.subplots()
Canvas_Result = FigureCanvasTkAgg(Fig_Result, master=center_pannel)
Canvas_Result.get_tk_widget().grid(row = 1, column = 0, padx = 5, pady = 5)

# Style
Fig_Result.set_facecolor(Theme['color_0'])
Fig_Result.get_axes()[0].set_facecolor(Theme['color_1'])
Fig_Result.get_axes()[0].spines['bottom'].set_color(Theme['color_3'])
Fig_Result.get_axes()[0].spines['top'].set_color(Theme['color_3'])
Fig_Result.get_axes()[0].spines['right'].set_color(Theme['color_3'])
Fig_Result.get_axes()[0].spines['left'].set_color(Theme['color_3'])
Fig_Result.get_axes()[0].xaxis.label.set_color(Theme['color_3'])
Fig_Result.get_axes()[0].yaxis.label.set_color(Theme['color_3'])
Fig_Result.get_axes()[0].tick_params(axis='x', colors=Theme['color_3'])
Fig_Result.get_axes()[0].tick_params(axis='y', colors=Theme['color_3'])

#================================================================================================================================================================================

# Run the Tkinter event loop
root.mainloop()
