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

from reconstruction import *
from libHSI.HSI_reconstruction import *


def loadSpectrum(filename):

    # Load spectral response from the system from .spd file
    return np.loadtxt(filename).astype(np.double)

target_name     = filedialog.askopenfilename()
samples_names   = filedialog.askopenfilenames()

T = np.array([loadSpectrum(target_name)])

print (T.shape)

Target, E_target = normalize_wavelength_sampling( T )

Samples = np.array([]).astype(np.double)

for sample_name in samples_names:

    Spectrum = loadSpectrum(sample_name)

    if np.size(Samples) == 0:
        Samples = np.array( [Spectrum] ).reshape( (Spectrum.shape[0], Spectrum.shape[1], 1) )
    else:
        Samples = np.append(Samples, Spectrum.reshape((Spectrum.shape[0], Spectrum.shape[1], 1)), axis=2)

Samples = np.array(Samples)
Samples, E_samples = normalize_wavelength_sampling(Samples)

####################################################################################################################

nsteps = 100; nrand = 10000

K_stocha_SAM = HSI_Rec_Stochastique( T, Samples, Criteria= "SAM", low_norm= 380, high_norm= 780, definition= 1, steps = nsteps, nrand = nrand )