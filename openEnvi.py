import numpy as np
import os
import re
import spectral.io.envi as envi

#-------------------------------------------------------------------------------------------------
def open_data(hdr_file, bin_file):

    header = envi.open(hdr_file, bin_file)
    cube = header.read_bands(header.bands.band_quantity)

    wavelengths = header.bands.centers

    return (cube, wavelengths)
#-------------------------------------------------------------------------------------------------
def get_spectrum(cube):

    cube_avg_rowwise = np.mean(cube, axis = 0)
    spectra = np.mean(cube_avg_rowwise, axis = 0)

    return spectra.flatten()
#-------------------------------------------------------------------------------------------------
def save_spd_data(spectrum, wavelength, filename = "spd_save.spd"):

    with open(filename, 'w') as file:
        for i in range(0, len(wavelength)):
            file.write(f"{wavelength[i]} {spectrum[i]}\n")
#-------------------------------------------------------------------------------------------------
def envi2spd(folder, save_folder):

    #for every hdr, get the corresponding bin and save the average spectrum
    files_hdr = [each for each in os.listdir(folder) if each.endswith('.hdr')]
    files_bin = [filename.replace(".hdr", ".bin") for filename in files_hdr]

    for i in range (0, len(files_hdr)):

        print(f"opening {folder + "/" + files_hdr[i]}")

        cube, wl = open_data(folder + "/" + files_hdr[i], folder + "/" + files_bin[i])
        spectrum = get_spectrum(cube)
        save_spd_data(spectrum, wl, save_folder + "/" + files_hdr[i].replace(".hdr", ".spd"))
#-------------------------------------------------------------------------------------------------
def envi2spd_normalized(folder, save_folder):

    # Does the same thing as envi2spd with the addition of a normalization step using black and white

    #for every hdr, get the corresponding bin and save the average spectrum
    files_black_hdr = [each for each in os.listdir(folder) if each.endswith('_Dark.hdr')]
    files_white_hdr = [filename.replace("_Dark.hdr", "_White.hdr") for filename in files_black_hdr]
    files_hdr = [filename.replace("_Dark.hdr", ".hdr") for filename in files_black_hdr]

    files_black_bin = [filename.replace("_Dark.hdr", "_Dark.bin") for filename in files_black_hdr]
    files_white_bin = [filename.replace("_Dark.hdr", "_White.bin") for filename in files_black_hdr]
    files_bin = [filename.replace("_Dark.hdr", ".bin") for filename in files_black_hdr]

    for i in range (0, len(files_hdr)):

        print(f"opening {folder + "/" + files_hdr[i]}")

        cube, wl = open_data(folder + "/" + files_hdr[i], folder + "/" + files_bin[i])
        cube_black, __ = open_data(folder + "/" + files_black_hdr[i], folder + "/" + files_black_bin[i])
        cube_white, __ = open_data(folder + "/" + files_white_hdr[i], folder + "/" + files_white_bin[i])
        spectrum = get_spectrum(cube)
        spectrum_black = get_spectrum(cube_black)
        spectrum_white = get_spectrum(cube_white)

        # Reflectance based on white and Dark reference
        spectrum_normalized = (spectrum - spectrum_black) * spectrum_white/spectrum

        save_spd_data(spectrum_normalized, wl, save_folder + "/" + files_hdr[i].replace(".hdr", ".spd"))
#-------------------------------------------------------------------------------------------------