# -*- coding: utf-8 -*-

from HSI_library import *
from HSI_distances import *

import spectral as spy

import time

#### Test parameters ####

test_count = 100000

#### Test Results ####

times = {}

#### Generate random data for testing ####

spectrum_set_A = np.abs(np.random.random((400, int(test_count / 10) ) ) ) * 100.0       # Intensity 100
spectrum_set_B = np.abs(np.random.random((400, int(test_count / 10) ) ) ) * 1000.0      # Intensity 1000

waves = np.arange(start= 380, step= 1, stop= 780)

#### TEST DISTANCES ####
timestamps = {}

###==========================================================================================================================================================================================###

timestamps['start_div_KL_HSI'] = time.time()

for k in range(0, test_count):
    i1 = np.random.randint(low = 0, high = 1000)
    i2 = np.random.randint(low = 0, high = 1000)
    r = div_KL(spectrum_set_A[:,i1], spectrum_set_B[:,i2], resolution= 1)

timestamps['end_div_KL_HSI'] = time.time()

timestamps['start_pseudo_div_KL_HSI'] = time.time()

times['avg_time_div_KL_HSI']             = (timestamps['end_div_KL_HSI']          - timestamps['start_div_KL_HSI'])                 / test_count

###==========================================================================================================================================================================================###

timestamps['start_dist_SAM_HSI'] = time.time()

for k in range(0, test_count):
    i1 = np.random.randint(low = 0, high = 1000)
    i2 = np.random.randint(low = 0, high = 1000)
    r = dist_SAM(spectrum_set_A[:,i1], spectrum_set_B[:,i2])

timestamps['end_dist_SAM_HSI'] = time.time()

timestamps['start_pseudo_dist_SAM_HSI'] = time.time()

times['avg_time_dist_SAM_HSI']             = (timestamps['end_dist_SAM_HSI']          - timestamps['start_dist_SAM_HSI'])           / test_count

###==========================================================================================================================================================================================###

timestamps['start_dist_SID_HSI'] = time.time()

for k in range(0, test_count):
    i1 = np.random.randint(low = 0, high = 1000)
    i2 = np.random.randint(low = 0, high = 1000)
    r = dist_SID(spectrum_set_A[:,i1], spectrum_set_B[:,i2])

timestamps['end_dist_SID_HSI'] = time.time()

timestamps['start_pseudo_dist_SID_HSI'] = time.time()

times['avg_time_dist_SID_HSI']             = (timestamps['end_dist_SID_HSI']          - timestamps['start_dist_SID_HSI'])           / test_count

###==========================================================================================================================================================================================###

timestamps['start_dist_SCA_HSI'] = time.time()

for k in range(0, test_count):
    i1 = np.random.randint(low = 0, high = 1000)
    i2 = np.random.randint(low = 0, high = 1000)
    r = dist_SCA(spectrum_set_A[:,i1], spectrum_set_B[:,i2])

timestamps['end_dist_SCA_HSI'] = time.time()

timestamps['start_pseudo_dist_SCA_HSI'] = time.time()

times['avg_time_dist_SCA_HSI']             = (timestamps['end_dist_SCA_HSI']          - timestamps['start_dist_SCA_HSI'])             / test_count

###==========================================================================================================================================================================================###

timestamps['start_dist_SGA_HSI'] = time.time()

for k in range(0, test_count):
    i1 = np.random.randint(low = 0, high = 1000)
    i2 = np.random.randint(low = 0, high = 1000)
    r = dist_SGA(spectrum_set_A[:,i1], spectrum_set_B[:,i2])

timestamps['end_dist_SGA_HSI'] = time.time()

timestamps['start_pseudo_dist_SGA_HSI'] = time.time()

times['avg_time_dist_SGA_HSI']             = (timestamps['end_dist_SGA_HSI']          - timestamps['start_dist_SGA_HSI'])             / test_count

###==========================================================================================================================================================================================###

timestamps['start_dist_euclidienne_cum_HSI'] = time.time()
diff = 0

for k in range(0, test_count):
    i1 = np.random.randint(low = 0, high = 1000)
    i2 = np.random.randint(low = 0, high = 1000)

    r = dist_euclidienne_cum(spectrum_set_A[:,i1], spectrum_set_B[:,i2])

print(diff)

timestamps['end_dist_euclidienne_cum_HSI'] = time.time()

timestamps['start_pseudo_dist_euclidienne_cum_HSI'] = time.time()

times['avg_time_dist_euclidienne_cum_HSI']             = (timestamps['end_dist_euclidienne_cum_HSI']          - timestamps['start_dist_euclidienne_cum_HSI'])             / test_count

###==========================================================================================================================================================================================###

timestamps['start_dist_euclidienne_cum_derive_HSI'] = time.time()

for k in range(0, test_count):
    i1 = np.random.randint(low = 0, high = 1000)
    i2 = np.random.randint(low = 0, high = 1000)
    r = dist_euclidienne_cum_derive(spectrum_set_A[:,i1], spectrum_set_B[:,i2])

timestamps['end_dist_euclidienne_cum_derive_HSI'] = time.time()

timestamps['start_pseudo_dist_euclidienne_cum_derive_HSI'] = time.time()

times['avg_time_dist_euclidienne_cum_derive_HSI']             = (timestamps['end_dist_euclidienne_cum_derive_HSI']          - timestamps['start_dist_euclidienne_cum_derive_HSI'])             / test_count

###==========================================================================================================================================================================================###

timestamps['start_dist_Minkowski_HSI'] = time.time()

for k in range(0, test_count):
    i1 = np.random.randint(low = 0, high = 1000)
    i2 = np.random.randint(low = 0, high = 1000)
    r = dist_Minkowski(spectrum_set_A[:,i1], spectrum_set_B[:,i2])

timestamps['end_dist_Minkowski_HSI'] = time.time()

timestamps['start_pseudo_dist_Minkowski_HSI'] = time.time()

times['avg_time_dist_Minkowski_HSI']             = (timestamps['end_dist_Minkowski_HSI']          - timestamps['start_dist_Minkowski_HSI'])             / test_count

###==========================================================================================================================================================================================###

timestamps['start_dist_bhatta_HSI'] = time.time()

for k in range(0, test_count):
    i1 = np.random.randint(low = 0, high = 1000)
    i2 = np.random.randint(low = 0, high = 1000)
    r = dist_bhatta(spectrum_set_A[:,i1], spectrum_set_B[:,i2])

timestamps['end_dist_bhatta_HSI'] = time.time()

timestamps['start_pseudo_dist_bhatta_HSI'] = time.time()

times['avg_time_dist_bhatta_HSI']             = (timestamps['end_dist_bhatta_HSI']          - timestamps['start_dist_bhatta_HSI'])             / test_count

###==========================================================================================================================================================================================###

timestamps['start_dist_chi_square_HSI'] = time.time()

for k in range(0, test_count):
    i1 = np.random.randint(low = 0, high = 1000)
    i2 = np.random.randint(low = 0, high = 1000)
    r = dist_chi_square(spectrum_set_A[:,i1], spectrum_set_B[:,i2], n = 2)

timestamps['end_dist_chi_square_HSI'] = time.time()

timestamps['start_pseudo_dist_chi_square_HSI'] = time.time()

times['avg_time_dist_chi_square_HSI']             = (timestamps['end_dist_chi_square_HSI']          - timestamps['start_dist_chi_square_HSI'])             / test_count

###==========================================================================================================================================================================================###

timestamps['start_dist_Csiszar_HSI'] = time.time()

for k in range(0, test_count):
    i1 = np.random.randint(low = 0, high = 1000)
    i2 = np.random.randint(low = 0, high = 1000)
    r = dist_Csiszar(spectrum_set_A[:,i1], spectrum_set_B[:,i2])

timestamps['end_dist_Csiszar_HSI'] = time.time()

timestamps['start_pseudo_dist_Csiszar_HSI'] = time.time()

times['avg_time_dist_Csiszar_HSI']             = (timestamps['end_dist_Csiszar_HSI']          - timestamps['start_dist_Csiszar_HSI'])             / test_count

###==========================================================================================================================================================================================###

timestamps['start_div_Csiszar_HSI'] = time.time()

for k in range(0, test_count):
    i1 = np.random.randint(low = 0, high = 1000)
    i2 = np.random.randint(low = 0, high = 1000)
    r = div_Csiszar(spectrum_set_A[:,i1], spectrum_set_B[:,i2])

timestamps['end_div_Csiszar_HSI'] = time.time()

timestamps['start_pseudo_div_Csiszar_HSI'] = time.time()

times['avg_time_div_Csiszar_HSI']             = (timestamps['end_div_Csiszar_HSI']          - timestamps['start_div_Csiszar_HSI'])             / test_count

###==========================================================================================================================================================================================###
"""
timestamps['start_pseudo_div_Csiszar2_HSI'] = time.time()

for k in range(0, test_count):
    i1 = np.random.randint(low = 0, high = 1000)
    i2 = np.random.randint(low = 0, high = 1000)
    r = pseudo_div_Csiszar2(spectrum_set_A[:,i1], spectrum_set_B[:,i2])

timestamps['end_pseudo_div_Csiszar2_HSI'] = time.time()

timestamps['start_pseudo_pseudo_div_Csiszar2_HSI'] = time.time()

times['avg_time_pseudo_div_Csiszar2_HSI']             = (timestamps['end_pseudo_div_Csiszar2_HSI']          - timestamps['start_pseudo_div_Csiszar2_HSI'])             / test_count
"""
###==========================================================================================================================================================================================###

timestamps['start_div_Hellinger_HSI'] = time.time()

for k in range(0, test_count):
    i1 = np.random.randint(low = 0, high = 1000)
    i2 = np.random.randint(low = 0, high = 1000)
    r = div_Hellinger(spectrum_set_A[:,i1], spectrum_set_B[:,i2])

timestamps['end_div_Hellinger_HSI']             = time.time()

timestamps['start_pseudo_div_Hellinger_HSI']    = time.time()

times['avg_time_div_Hellinger_HSI']             = (timestamps['end_div_Hellinger_HSI']          - timestamps['start_div_Hellinger_HSI'])             / test_count



"""
for k in range(0, 10000):
    i1 = np.random.randint(low = 0, high = 1000)
    i2 = np.random.randint(low = 0, high = 1000)
    r = pseudo_div_KL(spectrum_set_A[:,i1], spectrum_set_B[:,i2], resolution= 1)

timestamps['end_pseudo_div_KL'] = time.time()

timestamps['start_pseudo_div_KL2'] = time.time()

for k in range(0, 10000):
    i1 = np.random.randint(low = 0, high = 1000)
    i2 = np.random.randint(low = 0, high = 1000)
    r = pseudo_div_KL2(spectrum_set_A[:,i1], spectrum_set_B[:,i2], resolution= 1)

timestamps['end_pseudo_div_KL2'] = time.time()
"""


#times['avg_time_pseudo_div_KL']      = (timestamps['end_pseudo_div_KL']   - timestamps['start_pseudo_div_KL'])     / test_count
#times['avg_time_pseudo_div_KL2']     = (timestamps['end_pseudo_div_KL2']  - timestamps['start_pseudo_div_KL2'])    / test_count

for key in times.keys():
    print(f"{key} :\t\t{times[key]}")