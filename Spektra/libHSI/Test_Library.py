# -*- coding: utf-8 -*-

from HSI_library import *
import spectral as spy
import numpy as np

from tkinter import filedialog
import colour

#############################################################################################################

EPSILON = 10**(-6)

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

###### Compare RGB Reconstruction ######

#############################################################################################################
# Generate random image from list of normalized spectrums
#############################################################################################################
# Functions and I/O (load the spectrums, as spd values or npy)
#############################################################################################################

#interpolation function (Linear)
def linear_interpolation(data, nodes, t):
    """
    Linear Interpolation

    Parameters:
    - data:     NumPy array of float values.
    - nodes:    NumPy array of float values indicating the position of each point in the interval [0.0, 1.0].
    - t:        Value between 0 and 1 representing the position where interpolation is needed.

    Returns:
    - Interpolated value
    """

    n = len(nodes)
    result = 0.0

    for i in range(n-1):
        if (nodes[i] <= t):
            result = data[i] * abs( ( nodes[i+1] - t ) / ( nodes[i+1] - nodes[i] ) ) + data[i+1] * abs( ( nodes[i] - t ) / ( nodes[i+1] - nodes[i] ) )

    return result

#############################################################################################################

def load_spd(low_bound = 360.0, high_bound = 760.0, delta_lambda = 10):
    # Ask user to choose what spectrums to load
    filenames = filedialog.askopenfilenames()

    # Create the normalized wavelist
    new_waves   = np.arange(start= low_bound, stop= high_bound, step = delta_lambda)
    spectras    = np.zeros( ( len( filenames ), new_waves.size ) )

    # Load, Normalize and append data to the list
    for k, filename in enumerate(filenames):

        # 1 - Load the data from file
        data    = np.loadtxt(filename).astype(np.double)
        waves   = data[:, 0].flatten()
        values  = data[:, 1].flatten() 
        values /= np.max(values)

        # 2 - Normalize it using wavelist as the nodes point

        # A - Transforms Waves to nodes
        new_values  = np.zeros_like(new_waves)
        nodes = (waves - low_bound) * 1.0 / high_bound

        # B - Interpolate
        for i, t in enumerate( np.arange(start = 0.0, stop = 1.0, step = delta_lambda / (high_bound - low_bound) ) ):
            new_values[i] = linear_interpolation(values, nodes, t)

        spectras[k,:] = new_values
    
    return (spectras, new_waves)

#############################################################################################################

###### Generate random data ######

wd, hd = 128, 128
hbound, lbound = 750.0, 380.0
step_size = 1   # 1, 5, 10 or 20nm

Spectras, Wavelist = load_spd(low_bound= lbound, high_bound= hbound, delta_lambda= step_size)

Wavelist = np.arange(start = lbound, stop = hbound, step= step_size)
bd = len(Wavelist)

Cube = np.zeros( ( wd, hd, bd ) )

for x in range (0, wd):
    for y in range (0, hd):
        random_index    = np.random.randint( low = 0, high = Spectras.shape[0] )
        Cube[x,y,:]     = Spectras[random_index, :].flatten()

#############################################################################################################

def compareSpec_2_XYZ(Cube, Wavelist):

    XYZ_multi_hsi   = multiSpec2CMF_XYZ( Cube, Wavelist )
    XYZ_mono_hsi    = np.zeros_like( XYZ_multi_hsi )
    XYZ_mono_col    = np.zeros_like( XYZ_multi_hsi )

    for x in range(0, Cube.shape[0]):
        for y in range(0, Cube.shape[1]):

            dict = {}

            for i in range(0, len(Wavelist)):
                dict[Wavelist[i]] = Cube[x,y,i]

            XYZ_mono_hsi[x,y,:]     = monoSpec_2_CMF_XYZ(Cube[x,y,:], Wavelist)
            sd_sample               = colour.SpectralDistribution(dict, name = "sample")
            XYZ_mono_col[x,y,:]     = colour.sd_to_XYZ(sd_sample)

    XYZ_multi_hsi   /= np.max(XYZ_multi_hsi)
    XYZ_mono_hsi    /= np.max(XYZ_mono_hsi)
    XYZ_mono_col    /= np.max(XYZ_mono_col)

    delta_multi = XYZ_mono_col - XYZ_multi_hsi
    delta_mono  = XYZ_mono_col - XYZ_mono_hsi

    return (delta_multi, delta_mono)

#############################################################################################################

def compareSpec_2_RGB(Cube, Wavelist):

    RGB_multi_hsi   = multiSpec2CMF_RGB( Cube, Wavelist)
    RGB_mono_hsi    = np.zeros_like( RGB_multi_hsi )
    RGB_mono_col    = np.zeros_like( RGB_multi_hsi )

    for x in range(0, Cube.shape[0]):
        for y in range(0, Cube.shape[1]):

            dict = {}

            for i in range(0, len(Wavelist)):
                dict[Wavelist[i]] = Cube[x,y,i]

            RGB_mono_hsi[x,y,:]     = monoSpec2CMF_RGB(Cube[x,y,:], Wavelist)
            sd_sample               = colour.SpectralDistribution(dict, name = "sample")
            XYZ                     = colour.sd_to_XYZ(sd_sample)
            XYZ                    /= np.max(XYZ)
            RGB_mono_col[x,y,:]     = colour.XYZ_to_sRGB(XYZ)

    RGB_mono_col    = ( RGB_mono_col  - np.min( RGB_mono_col  ) ) / np.max( RGB_mono_col  - np.min( RGB_mono_col  ) )
    RGB_mono_hsi    = ( RGB_mono_hsi  - np.min( RGB_mono_hsi  ) ) / np.max( RGB_mono_hsi  - np.min( RGB_mono_hsi  ) )
    RGB_multi_hsi   = ( RGB_multi_hsi - np.min( RGB_multi_hsi ) ) / np.max( RGB_multi_hsi - np.min( RGB_multi_hsi ) )

    delta_multi = RGB_mono_col - RGB_multi_hsi
    delta_mono  = RGB_mono_col - RGB_mono_hsi

    return (delta_multi, delta_mono, RGB_mono_col, RGB_mono_hsi, RGB_multi_hsi)

#############################################################################################################

delta_multi_XYZ, delta_mono_XYZ = compareSpec_2_XYZ(Cube, Wavelist)
delta_multi_RGB, delta_mono_RGB, RGB_mono_col, RGB_mono_hsi, RGB_multi_hsi = compareSpec_2_RGB(Cube, Wavelist)

#############################################################################################################
# DELTA RGB Multi
#############################################################################################################

bound = max( np.abs( np.min( delta_multi_RGB ) ), np.max( delta_multi_RGB ) )

plt.imshow(delta_multi_RGB[:,:,0], cmap='bwr', vmin = -bound, vmax = bound)
plt.title('Difference reconstruction multi HSI et mono COLOURS cannal (R)')
plt.colorbar()

plt.show()

#############################################################################################################

plt.imshow(delta_multi_RGB[:,:,1], cmap='bwr', vmin = -bound, vmax = bound)
plt.title('Difference reconstruction multi HSI et mono COLOURS cannal (G)')
plt.colorbar()

plt.show()

#############################################################################################################

plt.imshow(delta_multi_RGB[:,:,2], cmap='bwr', vmin = -bound, vmax = bound)
plt.title('Difference reconstruction multi HSI et mono COLOURS cannal (B)')
plt.colorbar()

plt.show()

#############################################################################################################
# DELTA RGB Mono
#############################################################################################################

bound = max( np.abs( np.min( delta_mono_RGB ) ), np.max( delta_mono_RGB ) )

plt.imshow(delta_mono_RGB[:,:,0], cmap='bwr', vmin = -bound, vmax = bound)
plt.title('Difference reconstruction mono HSI et mono COLOURS cannal (R)')
plt.colorbar()

plt.show()

#############################################################################################################

plt.imshow(delta_mono_RGB[:,:,1], cmap='bwr', vmin = -bound, vmax = bound)
plt.title('Difference reconstruction mono HSI et mono COLOURS cannal (G)')
plt.colorbar()

plt.show()

#############################################################################################################

plt.imshow(delta_mono_RGB[:,:,2], cmap='bwr', vmin = -bound, vmax = bound)
plt.title('Difference reconstruction mono HSI et mono COLOURS cannal (B)')
plt.colorbar()

plt.show()

#############################################################################################################
# DELTA XYZ Multi
#############################################################################################################

bound = max( np.abs( np.min( delta_multi_XYZ ) ), np.max( delta_multi_XYZ ) )

plt.imshow(delta_multi_XYZ[:,:,0], cmap='bwr', vmin = -bound, vmax = bound)
plt.title('Difference reconstruction multi HSI et mono COLOURS cannal (X)')
plt.colorbar()

plt.show()

#############################################################################################################

plt.imshow(delta_multi_XYZ[:,:,1], cmap='bwr', vmin = -bound, vmax = bound)
plt.title('Difference reconstruction multi HSI et mono COLOURS cannal (Y)')
plt.colorbar()

plt.show()

#############################################################################################################

plt.imshow(delta_multi_XYZ[:,:,2], cmap='bwr', vmin = -bound, vmax = bound)
plt.title('Difference reconstruction multi HSI et mono COLOURS cannal (Z)')
plt.colorbar()

plt.show()

#############################################################################################################
# DELTA XYZ Mono
#############################################################################################################

bound = max( np.abs( np.min( delta_mono_XYZ ) ), np.max( delta_mono_XYZ ) )

plt.imshow(delta_mono_XYZ[:,:,0], cmap='bwr', vmin = -bound, vmax = bound)
plt.title('Difference reconstruction mono HSI et mono COLOURS cannal (X)')
plt.colorbar()

plt.show()

#############################################################################################################

plt.imshow(delta_mono_XYZ[:,:,1], cmap='bwr', vmin = -bound, vmax = bound)
plt.title('Difference reconstruction mono HSI et mono COLOURS cannal (Y)')
plt.colorbar()

plt.show()

#############################################################################################################

plt.imshow(delta_mono_XYZ[:,:,2], cmap='bwr', vmin = -bound, vmax = bound)
plt.title('Difference reconstruction mono HSI et mono COLOURS cannal (Z)')
plt.colorbar()

plt.show()

#############################################################################################################
# Result
#############################################################################################################

plt.imshow(RGB_multi_hsi)
plt.title('Reconstruction RGB multi HSI')

plt.show()

#############################################################################################################

plt.imshow(RGB_mono_col)
plt.title('Reconstruction RGB mono Colours')

plt.show()

#############################################################################################################

plt.imshow(RGB_mono_hsi)
plt.title('Reconstruction RGB mono HSI')

plt.show()