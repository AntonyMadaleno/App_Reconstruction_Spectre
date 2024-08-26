import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog as fd

from libHSI.HSI_reconstruction import *
from libHSI.HSI_distances import *

from reconstruction import *

### LOAD LIGHTING SPECTRUM ###

spd, Energy = normalize_wavelength_sampling( np.array( [np.loadtxt(fd.askopenfilename()).astype(np.double)] ) )

Waves   = spd[0][0]
D65     = spd[0][1]

### Load the filters spectrums ###
liste_norm, Es = normalize_wavelength_sampling( np.array( [np.loadtxt(fd.askopenfilename()).astype(np.double)] ) )

spectra = (liste_norm[0][1] * Es / 100) * D65 * Energy

np.savetxt(fd.asksaveasfilename(), np.array([Waves, spectra]).T )

print(f"Energy D65 : {Energy[0]}")
print(f"Energy F.. : {np.trapz(spectra, dx = 1.0)}")
print(f"average filter : {np.average(liste_norm[0][1] * Es / 100)}")
