# IMPORTS ##########################################################################################################################
from tkinter import filedialog as fd
import numpy as np
from reconstruction import normalize_wavelength_sampling

#Global parameters
LB = 380
HB = 760

# Charger les spectres d'éclairement des leds

filenames = fd.askopenfilenames(defaultextension='spd')

led_Data = []

for filename in filenames:
    led_Data.append( np.loadtxt(filename).astype(np.double) )

# Normaliser les spectres d'éclairement
Normalized_Leds_Data, Led_Energie = normalize_wavelength_sampling(np.array(led_Data), definition=1.0, low_bound= LB, high_bound= HB)

Waves = Normalized_Leds_Data[0][0]
Normalized_Leds_Intensities = []

for i in range( 0, len(Normalized_Leds_Data) ):
    Normalized_Leds_Intensities.append( Normalized_Leds_Data[i][1] )

# Charger le spectre cible à reconstruire

filename = fd.askopenfilename(defaultextension='spd')
target_Data = np.loadtxt(filename).astype(np.double)

# Normaliser les spectres d'éclairement
Normalized_Target_Data, Target_Energie = normalize_wavelength_sampling(np.array([target_Data]), definition=1.0, low_bound= LB, high_bound= HB)
Target_Energie = Target_Energie[0]
Normalized_Target_Intensities = Normalized_Target_Data[0][1]

# Paramètre de la boucle d'optimisation

population      = 20000
generation      = 200
mutation_rate   = 0.5
selection_per   = 0.05

print(len(Normalized_Leds_Intensities))

# Cacul des coefficients

# Sauvegarde des divers résultats