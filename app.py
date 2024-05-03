import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import Tk, Label, Entry, Button, Frame, OptionMenu, StringVar, Listbox, END, Scrollbar, Checkbutton, IntVar, RIGHT, filedialog, colorchooser
import pandas as pd
import os

from reconstruction import *

definition = 1000
curves = []  # List to store curve information: (name, x, y, color)
main_curve = []

xlims = (300.0, 1000.0)

#========================================================================================================================================================================================================
def gen_color(spec, waves):

    XYZ = monoSpec_2_CMF_XYZ(spec, waves, cmf = 'libHSI/data/#_360_830_2deg.npy')
    x = XYZ[0,0] / np.sum(XYZ)
    y = XYZ[0,1] / np.sum(XYZ)
    RGB = (monoXYZ_2_RGB(XYZ, RGB_space="sRGB", Gamma= 1.0) * 255.0).astype(np.uint8)
    # Convert RGB to hexadecimal color code
    hex_color = '#%02x%02x%02x' % (RGB[0,0], RGB[0,1], RGB[0,2])

    return hex_color
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
    waves = data[:, 0]
    spec = data[:, 1]
    return name, waves, spec
#========================================================================================================================================================================================================
def plot_curves():

    global xlims

    ax.clear()
    for curve in curves:
        name, x, y, color = curve
        ax.plot(x, y, color=color, alpha = 0.5, lw = 3, label=name)

    ax.set_xlabel('wavelength (nm)', color=base_color)
    ax.set_ylabel('Y', color=base_color)
    ax.set_title('Intensity per wavelength', color=base_color)
    ax.grid(True)

    xlims = fig.get_axes()[0].get_xlim()
    fig_main.get_axes()[0].set_xlim(xlims)
    plt.tight_layout()
    canvas.draw()
#========================================================================================================================================================================================================
def plot_selected(index_main = None):
    global xlims

    ax_main.clear()

    if len(main_curve) > 0:
        name, x, y, color = main_curve[0]
        ax_main.plot(x, y, color=color, alpha = 0.5, lw = 3, label=name)

    ax_main.set_xlabel('wavelength (nm)', color=base_color)
    ax_main.set_ylabel('Y', color=base_color)
    ax_main.set_title('Intensity per wavelength', color=base_color)
    ax_main.grid(True)

    fig_main.get_axes()[0].set_xlim(xlims)
    plt.tight_layout()
    canvas_main.draw()
#========================================================================================================================================================================================================
def add_curve():
    name = curve_name_entry.get()
    lower_bound = float(lower_bound_entry.get())
    higher_bound = float(upper_bound_entry.get())
    center = float(center_entry.get())
    fwhm = float(fwhm_entry.get())
    x = np.linspace(lower_bound, higher_bound, definition)
    y = gaussian_distribution(x, center, fwhm, float(coeff_entry.get()))

    color = gen_color(x,y)

    curves.append((name, x, y, color))

    plot_curves()
    update_curve_list()
#========================================================================================================================================================================================================
def load_curves():
    folder_path = filedialog.askdirectory()
    if folder_path:
        spd_files = [file for file in os.listdir(folder_path) if file.endswith('.spd')]
        for spd_file in spd_files:
            file_path = os.path.join(folder_path, spd_file)
            name, waves, spec = load_spd_file(file_path)
            color = gen_color(spec, waves)
            curves.append((name, waves, spec, color))

        plot_curves()
        update_curve_list()
#========================================================================================================================================================================================================
def delete_curves():
    indexes = curve_list.curselection()
    indexes = [int(index) for index in indexes]
    indexes.sort(reverse=True)
    for index in indexes:
        curves.pop(index)
    plot_curves()
    update_curve_list()
#========================================================================================================================================================================================================
def select_main_curves():
    index = curve_list.curselection()[0]

    if (len(main_curve) > 0):
        main_curve.pop()

    main_curve.append(curves.pop(index))
    plot_curves()
    plot_selected()
    update_curve_list()
#========================================================================================================================================================================================================
def fit_main_curves():

    coefficients = None
    coef_norm = None
    rest = None

    if (method.get() == "default"):
        coefficients, rest = curve_rectruction(main_curve[0], curves, float(low_bound_norm_entry.get()), float(high_bound_norm_entry.get()) , float(definition_norm_entry.get()) )
        coef_norm = coefficients / np.max(coefficients)
        return
    
    elif(method.get() == "Scalar"):
        coefficients, rest = curve_rectruction_scalar(main_curve[0], curves, float(low_bound_norm_entry.get()), float(high_bound_norm_entry.get()) , float(definition_norm_entry.get()) )
        coef_norm = coefficients / np.max(coefficients)
        return


#========================================================================================================================================================================================================
def update_curve_list():
    curve_list.delete(0, END)
    for i, curve in enumerate(curves):
        name, _, _, _ = curve
        curve_list.insert(END, name)
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
input_gaussian_parameters = Frame(left_pannel, bg = dark_color)
input_gaussian_parameters.pack(pady=5, side=tk.TOP)

# Create labels and entry widgets for user inputs for curve parameters
Label(input_gaussian_parameters, text="Curve Name:", font=default_font, foreground= light_color, bg = dark_color).grid(row=0, column=0, sticky='w', padx=5, pady=5)
curve_name_entry = Entry(input_gaussian_parameters, font=default_font, foreground= light_color, bg = dark_color)
curve_name_entry.grid(row=0, column=1, padx=5, pady=5)
curve_name_entry.insert(0, "default_name")  # Default value

Label(input_gaussian_parameters, text="Lower Bound (nm):", font=default_font, foreground= light_color, bg = dark_color).grid(row=1, column=0, sticky='w', padx=5, pady=5)
lower_bound_entry = Entry(input_gaussian_parameters, font=default_font, foreground= light_color, bg = dark_color)
lower_bound_entry.grid(row=1, column=1, padx=5, pady=5)
lower_bound_entry.insert(400, "400")  # Default value

Label(input_gaussian_parameters, text="Upper Bound (nm):", font=default_font, foreground= light_color, bg = dark_color).grid(row=2, column=0, sticky='w', padx=5, pady=5)
upper_bound_entry = Entry(input_gaussian_parameters, font=default_font, foreground= light_color, bg = dark_color)
upper_bound_entry.grid(row=2, column=1, padx=5, pady=5)
upper_bound_entry.insert(800, "800")  # Default value

Label(input_gaussian_parameters, text="Center (nm):", font=default_font, foreground= light_color, bg = dark_color).grid(row=3, column=0, sticky='w', padx=5, pady=5)
center_entry = Entry(input_gaussian_parameters, font=default_font, foreground= light_color, bg = dark_color)
center_entry.grid(row=3, column=1, padx=5, pady=5)
center_entry.insert(600, "600")  # Default value

Label(input_gaussian_parameters, text="FWHM (nm):", font=default_font, foreground= light_color, bg = dark_color).grid(row=4, column=0, sticky='w', padx=5, pady=5)
fwhm_entry = Entry(input_gaussian_parameters, font=default_font, foreground= light_color, bg = dark_color)
fwhm_entry.grid(row=4, column=1, padx=5, pady=5)
fwhm_entry.insert(20, "20")  # Default value

Label(input_gaussian_parameters, text="Coefficient:", font=default_font, foreground= light_color, bg = dark_color).grid(row=5, column=0, sticky='w', padx=5, pady=5)
coeff_entry = Entry(input_gaussian_parameters, font=default_font, foreground= light_color, bg = dark_color)
coeff_entry.grid(row=5, column=1, padx=5, pady=5)
coeff_entry.insert(1, "1.0")  # Default value

# Create buttons for adding curves
add_button = Button(input_gaussian_parameters, text="Add Curve", command=add_curve, font=default_font, relief = "flat", bg = base_color, activebackground = "#33cccc")
add_button.grid(row=7, column=0, columnspan = 1, padx=5, pady=5)

### LOAD / DELETE Curves ###
# Create buttons for adding curves
load_button = Button(input_gaussian_parameters, text="Load Curves (spd)", command=load_curves, font=default_font, relief = "flat", bg = base_color, activebackground = "#33cccc")
load_button.grid(row = 7, column = 1, columnspan = 1, padx=5, pady=5)

# Create a frame for the curve list
curve_list_frame = Frame(left_pannel, bg = dark_color)
curve_list_frame.pack(pady=10, padx=3, side='bottom')

# Create a label for the curve list
Label(curve_list_frame, text="Curves:", font=default_font, foreground= light_color, bg = dark_color).pack()

# Create a listbox for displaying curve names with checkboxes
curve_list = Listbox(curve_list_frame, selectmode="multiple", font=default_font, foreground= light_color, bg = light_dark_color, width=40)
curve_list.pack(pady=10, padx = 3)

scrollbar = Scrollbar(curve_list_frame, orient="vertical")
scrollbar.config(command=curve_list.yview)
scrollbar.pack(side="right", fill="y")

curve_list.config(yscrollcommand=scrollbar.set)

# Create a button to delete selected curves
delete_button = Button(curve_list_frame, text="Delete Selected Curves", command=delete_curves, font=default_font, relief = "flat", bg = base_color, activebackground = "#33cccc")
delete_button.pack(pady=5)

### Fitting parameters ###

fitting_inputs = Frame(left_pannel, bg = dark_color)
fitting_inputs.pack(pady=10, padx = 3, side=tk.BOTTOM)

# Create a button to select the curve to fit to selected curves
select_button = Button(curve_list_frame, text="Select Main Curve", command=select_main_curves, font=default_font, relief = "flat", bg = base_color, activebackground = "#33cccc")
select_button.pack(pady=5)

Label(fitting_inputs, text="low bound for normalization (nm):", font=default_font, foreground= light_color, bg = dark_color).grid(row=0, column=0, sticky='w', padx=5, pady=5)
low_bound_norm_entry = Entry(fitting_inputs, font=default_font, foreground= light_color, bg = dark_color)
low_bound_norm_entry.grid(row=0, column=1, padx=5, pady=5)
low_bound_norm_entry.insert(300, "300.0")  # Default value

Label(fitting_inputs, text="high bound for normalization (nm):", font=default_font, foreground= light_color, bg = dark_color).grid(row=1, column=0, sticky='w', padx=5, pady=5)
high_bound_norm_entry = Entry(fitting_inputs, font=default_font, foreground= light_color, bg = dark_color)
high_bound_norm_entry.grid(row=1, column=1, padx=5, pady=5)
high_bound_norm_entry.insert(1000, "1000.0")  # Default value

Label(fitting_inputs, text="definition for normalization (nm):", font=default_font, foreground= light_color, bg = dark_color).grid(row=2, column=0, sticky='w', padx=5, pady=5)
definition_norm_entry = Entry(fitting_inputs, font=default_font, foreground= light_color, bg = dark_color)
definition_norm_entry.grid(row=2, column=1, padx=5, pady=5)
definition_norm_entry.insert(1, "1.0")  # Default value

# Create a button to fit the curve
fit_button = Button(fitting_inputs, text="Curve fitting", command=fit_main_curves, font=default_font, relief = "flat", bg = base_color, activebackground = "#33cccc")
fit_button.grid(row = 3, column = 0, columnspan = 2, padx= 5, pady= 5)

# Create a option menu to choose fitting optimisation
methods_names = ["Default", "Scalar", "SMA"]

method = StringVar()
method.set(methods_names[0])

method_menu = OptionMenu(fitting_inputs, method, *methods_names)
method_menu.grid(row = 4, rowspan = 3, column= 0, columnspan= 2, padx= 5, pady= 5)

### GRAPH PLOTS ###

plots = Frame(root, bg = "#121212")
plots.pack(pady = 0, padx= 3, side= tk.RIGHT, fill='both', expand= True)

# Create a Matplotlib figure and canvas for displaying the plot
fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=plots)
canvas.get_tk_widget().pack(padx= 5, pady= 5, side='top', fill='both', expand=True)

# Create a Matplotlib figure and canvas for displaying the plot
fig_main, ax_main = plt.subplots()
canvas_main = FigureCanvasTkAgg(fig_main, master=plots)
canvas_main.get_tk_widget().pack(padx= 5, pady= 5, side='bottom', fill='both', expand=True)

# Style
fig.set_facecolor("#121212")
fig.get_axes()[0].set_facecolor(dark_color)
fig.get_axes()[0].spines['bottom'].set_color(light_color)
fig.get_axes()[0].spines['top'].set_color(light_color)
fig.get_axes()[0].spines['right'].set_color(light_color)
fig.get_axes()[0].spines['left'].set_color(light_color)
fig.get_axes()[0].xaxis.label.set_color(light_color)
fig.get_axes()[0].yaxis.label.set_color(light_color)
fig.get_axes()[0].tick_params(axis='x', colors=light_color)
fig.get_axes()[0].tick_params(axis='y', colors=light_color)

fig_main.set_facecolor("#121212")
fig_main.get_axes()[0].set_facecolor(dark_color)
fig_main.get_axes()[0].spines['bottom'].set_color(light_color)
fig_main.get_axes()[0].spines['top'].set_color(light_color)
fig_main.get_axes()[0].spines['right'].set_color(light_color)
fig_main.get_axes()[0].spines['left'].set_color(light_color)
fig_main.get_axes()[0].xaxis.label.set_color(light_color)
fig_main.get_axes()[0].yaxis.label.set_color(light_color)
fig_main.get_axes()[0].tick_params(axis='x', colors=light_color)
fig_main.get_axes()[0].tick_params(axis='y', colors=light_color)

# Initialize the plot and curve list
plot_curves()
plot_selected()
update_curve_list()

# Run the Tkinter event loop
root.mainloop()
