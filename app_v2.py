import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import Tk, Label, Entry, Button, Frame, OptionMenu, StringVar, Listbox, END, Scrollbar, Checkbutton, IntVar, RIGHT, filedialog, colorchooser
import pandas as pd
import cv2
from PIL import Image, ImageTk
import os

from libHSI.HSI_library import *

from reconstruction import *

try:
    bands = []  # List to store curve information: (name, x, y, color)
    XYZ = []
    #========================================================================================================================================================================================================
    def fwhm2std_dev(fwhm):
        return float(fwhm) / 2.355
    #========================================================================================================================================================================================================
    def gaussian_distribution(x, c, fwhm, k = 1.0):
        return k * np.exp(-(x - c)**2 / (2 * fwhm2std_dev(fwhm)**2))
    #========================================================================================================================================================================================================
    def load_spd_file(file_path):
        # Load data from .spd file
        data = np.loadtxt(file_path)
        name = file_path.split('/')[-1].split('.')[0]  # Extract file name as curve name
        x = data[:, 0]
        y = data[:, 1]
        return name, x, y
    #========================================================================================================================================================================================================
    def load_image():
        img_path = filedialog.askopenfilename()
        spd_path = filedialog.askopenfilename()
        name, x, y = load_spd_file(spd_path)
        data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        bands.append( [name, x, y, data] )
        update_image_list()
    #========================================================================================================================================================================================================
    def delete_bands():
        indexes = image_list.curselection()
        indexes = [int(index) for index in indexes]
        indexes.sort(reverse=True)
        for index in indexes:
            bands.pop(index)
        update_image_list()
    #========================================================================================================================================================================================================
    def fit_main_curves(WL, target, curves):

        cmf = ("target", WL, target, light_color)

        coefficients, rest = curve_rectruction(cmf, curves, float(np.min(WL)), float(np.max(WL)) , float(np.abs(WL[1]- WL[0])) )

        print(coefficients)
        print(coefficients.shape)
        coef_norm = coefficients / np.max(coefficients)

        return coef_norm
    #========================================================================================================================================================================================================
    def update_image_list():
        image_list.delete(0, END)
        for i, band in enumerate(bands):
            name = band[0]
            image_list.insert(END, name)
    #========================================================================================================================================================================================================
    def load_cmf():
        # open dialogue to choose the file (CSV)
        filename = filedialog.askopenfilename()

        # read file and transfer its data as numpy arrays (WL, X, Y, Z) XYZ not necessary it depend on the colorspace of the cmf
        data = pd.read_csv(filename).to_numpy()
        WL, X, Y, Z = data[:,0], data[:,1], data[:,2], data[:,3]

        XYZ = np.zeros((X.shape[0], 3))
        XYZ[:,0], XYZ[:,1], XYZ[:,2] = X, Y, Z

        return WL, XYZ
    #========================================================================================================================================================================================================
    """
    CIE RGB Reference E
    2.3706743 -0.9000405 -0.4706338
    -0.5138850  1.4253036  0.0885814
    0.0052982 -0.0146949  1.0093968
    """
    def xyz2rgb(XYZ):
        M = np.array([
            [2.3706743, -0.9000405, -0.4706338],
            [-0.5138850,  1.4253036,  0.0885814],
            [0.0052982, -0.0146949,  1.0093968]
        ])
        
        return np.dot(M, XYZ)
    #========================================================================================================================================================================================================
    """def RGB_reconstruction():

        Chroma = np.array(
            [
                [0.6400, 0.3300],
                [0.3000, 0.6000],
                [0.1500, 0.0600]
            ]
        )

        RW = np.array([0.95047, 1.00000, 1.08883]) #D65

        curves = []
        datas = []

        for band in bands:
            curves.append((band[0], band[1], band[2], base_color))
            datas.append(band[3])

        WL, XYZ_curves = load_cmf()

        KR = fit_main_curves(WL, XYZ_curves[:,0], curves)
        KG = fit_main_curves(WL, XYZ_curves[:,1], curves)
        KB = fit_main_curves(WL, XYZ_curves[:,2], curves)

        xyz = np.zeros((datas[0].shape[0], datas[0].shape[1], 3)).astype(float)

        for i in range(np.size(KR)):
            xyz[:,:,0] += KR[i] * datas[i]
            xyz[:,:,1] += KG[i] * datas[i]
            xyz[:,:,2] += KB[i] * datas[i]

        M = Calc_XYZ2RGB_Matrix(Chroma, RW)

        RGB = (XYZ_2_RGB_space(xyz, M, Gamma = float( gamma_entry.get() )) * 255.0).astype(np.uint8)

        im = Image.fromarray(RGB)
        photo = ImageTk.PhotoImage(image=im)
        result_display.config(image=photo)
        result_display.image = photo

        root.update()

        return"""
    
    def RGB_reconstruction():

        Chroma = np.array(
            [
                [0.6400, 0.3300],
                [0.3000, 0.6000],
                [0.1500, 0.0600]
            ]
        )

        RW = np.array([0.95047, 1.00000, 1.08883]) #D65

        curves = []
        datas = []

        for band in bands:
            curves.append((band[0], band[1], band[2], base_color))
            datas.append(band[3])

        WL, XYZ_curves = load_cmf()

        KR = fit_main_curves(WL, XYZ_curves[:,0], curves)
        KG = fit_main_curves(WL, XYZ_curves[:,1], curves)
        KB = fit_main_curves(WL, XYZ_curves[:,2], curves)

        xyz = np.zeros((datas[0].shape[0], datas[0].shape[1], 3)).astype(float)

        for i in range(np.size(KR)):
            xyz[:,:,0] += KR[i] * datas[i]
            xyz[:,:,1] += KG[i] * datas[i]
            xyz[:,:,2] += KB[i] * datas[i]

        #M = Calc_XYZ2RGB_Matrix(Chroma, RW)

        #RGB = (XYZ_2_RGB_space(xyz, M, Gamma = float( gamma_entry.get() )) * 255.0).astype(np.uint8)

        g = 1.0/ float(gamma_entry.get())

        xyz[:,0] = (xyz[:,0]**g) * 255 / np.max(xyz[:,0])
        xyz[:,1] = (xyz[:,1]**g) * 255 / np.max(xyz[:,1])
        xyz[:,2] = (xyz[:,2]**g) * 255 / np.max(xyz[:,2])

        xyz = np.clip(xyz, a_max= 255, a_min=0)

        im = Image.fromarray(xyz.astype(np.uint8))
        photo = ImageTk.PhotoImage(image=im)
        result_display.config(image=photo)
        result_display.image = photo

        root.update()

        return
    #========================================================================================================================================================================================================
    # ============== GUI TKINTER ================#
    #========================================================================================================================================================================================================

    # Create a Tkinter window
    root = Tk()
    root.configure(background= "#121212")

    # Set default font to Segoe UI
    default_font = ("Segoe UI", 12)

    dark_color = "#202020"
    light_dark_color = "#323232"
    light_color = "#f1f1f1"
    base_color = "#00cc66"

    ### LEFT INPUTS PANNEL ###

    left_pannel = Frame(root, bg = dark_color)
    left_pannel.pack(pady = 10, padx= 3, side= tk.LEFT)

    ### GAUSSIAN PARAMETERS INPUTS ###

    # Create frames for the input/output fields
    input_data_loading = Frame(left_pannel, bg = dark_color)
    input_data_loading.pack(pady=5, side=tk.TOP)

    ### LOAD / DELETE Curves ###
    # Create buttons for adding curves
    load_button = Button(input_data_loading, text="Load Image", command=load_image, font=default_font, relief = "flat", bg = base_color, activebackground = "#33cccc")
    load_button.grid(row = 7, column = 1, columnspan = 1, padx=5, pady=5)

    # Create a frame for the curve list
    image_list_frame = Frame(left_pannel, bg = dark_color)
    image_list_frame.pack(pady=10, padx=3, side='bottom')

    # Create a label for the curve list
    Label(image_list_frame, text="Images loaded:", font=default_font, foreground= light_color, bg = dark_color).pack()

    # Create a listbox for displaying curve names with checkboxes
    image_list = Listbox(image_list_frame, selectmode="multiple", font=default_font, foreground= light_color, bg = light_dark_color, width=40)
    image_list.pack(pady=10, padx = 3)

    scrollbar = Scrollbar(image_list_frame, orient="vertical")
    scrollbar.config(command=image_list.yview)
    scrollbar.pack(side="right", fill="y")

    image_list.config(yscrollcommand=scrollbar.set)

    # Create a button to delete selected curves
    delete_button = Button(image_list_frame, text="Delete Selected Bands", command=delete_bands, font=default_font, relief = "flat", bg = base_color, activebackground = "#33cccc")
    delete_button.pack(pady=5)

    ### Fitting parameters ###

    rebuild_inputs = Frame(left_pannel, bg = dark_color)
    rebuild_inputs.pack(pady=10, padx = 3, side=tk.BOTTOM)

    ### Image Display ###

    display_pannel = Frame(root, bg = dark_color)
    display_pannel.pack(pady = 10, padx= 3, side= tk.RIGHT, fill='both', expand=True)

    result_display = Label(display_pannel, bg="#121212")
    result_display.pack(pady=5, padx=5, fill='both', expand=True)

    # Create a button to fit the curve
    fit_button = Button(rebuild_inputs, text="RGB Reconstruction", command=RGB_reconstruction, font=default_font, relief = "flat", bg = base_color, activebackground = "#33cccc")
    fit_button.grid(row = 3, column = 0, columnspan = 2, padx= 5, pady= 5)

    #Create an input for gamma parameter
    Label(rebuild_inputs, text="Gamma (2.2 for D65):", font=default_font, foreground= light_color, bg = dark_color).grid(row=0, column=0, sticky='w', padx=5, pady=5)
    gamma_entry = Entry(rebuild_inputs, font=default_font, foreground= light_color, bg = dark_color)
    gamma_entry.grid(row=0, column=1, padx=5, pady=5)
    gamma_entry.insert(0, "2.2")  # Default value

    # Run the Tkinter event loop

    root.mainloop()
except Exception as e:
    print("An error occurred:", e)
