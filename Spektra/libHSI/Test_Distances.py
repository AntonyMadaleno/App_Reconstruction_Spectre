# -*- coding: utf-8 -*-

from HSI_library import *
from HSI_distances import *

import time

#### Test parameters ####

test_count = 10000

#### Test Results ####

times = {}

#### Generate random data for testing ####

spectrum_set_A = np.abs(np.random.random((400,1000))) * 100.0       # Intensity 100
spectrum_set_B = np.abs(np.random.random((400,1000))) * 1000.0      # Intensity 1000

waves = np.arange(start= 380, step= 1, stop= 780)

#### TEST DISTANCES ####
timestamps = {}

timestamps['start_divKL'] = time.time()

for k in range(0, 10000):
    i1 = np.random.randint(low = 0, high = 1000)
    i2 = np.random.randint(low = 0, high = 1000)
    r = div_KL(spectrum_set_A[:,i1], spectrum_set_B[:,i2], resolution= 1)

timestamps['end_divKL'] = time.time()

timestamps['start_pseudo_divKL'] = time.time()

for k in range(0, 10000):
    i1 = np.random.randint(low = 0, high = 1000)
    i2 = np.random.randint(low = 0, high = 1000)
    r = pseudo_div_KL(spectrum_set_A[:,i1], spectrum_set_B[:,i2], resolution= 1)

timestamps['end_pseudo_divKL'] = time.time()

timestamps['start_pseudo_divKL2'] = time.time()

for k in range(0, 10000):
    i1 = np.random.randint(low = 0, high = 1000)
    i2 = np.random.randint(low = 0, high = 1000)
    r = pseudo_div_KL2(spectrum_set_A[:,i1], spectrum_set_B[:,i2], resolution= 1)

timestamps['end_pseudo_divKL2'] = time.time()



times['avg_time_divKL']             = (timestamps['end_divKL']          - timestamps['start_divKL'])            / test_count
times['avg_time_pseudo_divKL']      = (timestamps['end_pseudo_divKL']   - timestamps['start_pseudo_divKL'])     / test_count
times['avg_time_pseudo_divKL2']     = (timestamps['end_pseudo_divKL2']  - timestamps['start_pseudo_divKL2'])    / test_count

print(times)