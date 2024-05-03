# -*- coding: utf-8 -*-
"""
Extrait les données des illuminants et des colour matching function CIE-XYZ à 2°
Created on Mon Aug 31 10:49:39 2015
Released on 15/10/2016

@author: martin, noel
"""
import matplotlib.pyplot as plt
import openpyxl
import numpy as np

##============================================================================
##========================= les illuminants ==================================
##============================================================================
wavelength = np.zeros(97)
D65 = np.zeros(97)
D50 = np.zeros(97)
E = np.zeros(97)
A = np.zeros(97)

excel_classeur = openpyxl.load_workbook('illuminant.xlsx',keep_vba=True)
excel_feuille  = excel_classeur[ 'StOllum-ID' ]

for lg in range (97):
   wavelength[lg] = excel_feuille['A'+str(lg+2)].value
   D65[lg] = excel_feuille['B'+str(lg+2)].value
   D50[lg] = excel_feuille['C'+str(lg+2)].value
   E[lg] = excel_feuille['F'+str(lg+2)].value
   A[lg] = excel_feuille['G'+str(lg+2)].value

np.save('I_D65_300-780.npy',D65)
np.save('I_D50_300-780.npy',D50)
np.save('I_E_300-780.npy',E)
np.save('I_A_300-780.npy',A)
np.save('I_wavelength_300-780.npy',wavelength)

##============================================================================
##========================= les CMF CIE-XYZ 2° ===============================
##============================================================================
wavelength = np.zeros(95)
X = np.zeros(95)
Y = np.zeros(95)
Z = np.zeros(95)

excel_classeur = openpyxl.load_workbook('physiologic_curves.xlsx',keep_vba=True)
excel_feuille  = excel_classeur[ 'XYZ_curves_2' ]

for lg in range (95):
    wavelength[lg] = excel_feuille['A'+str(lg+2)].value    
    X[lg] = excel_feuille['B'+str(lg+2)].value
    Y[lg] = excel_feuille['C'+str(lg+2)].value
    Z[lg] = excel_feuille['D'+str(lg+2)].value

np.save('X_360_830_2deg.npy',X)
np.save('Y_360_830_2deg.npy',Y)
np.save('Z_360_830_2deg.npy',Z)

plt.figure()
plt.plot(w,D50,label='D50')
plt.plot(w,D65,label='D65')
plt.plot(w,E,label='E')
plt.plot(w,A,label='A')

plt.legend()
plt.xlabel('Wavelength')
