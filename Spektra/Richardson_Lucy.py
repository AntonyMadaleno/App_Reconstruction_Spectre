import numpy as np
import cv2
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

from Recalage import brenner
from tqdm import tqdm

from tkinter import filedialog

#================================================================================================================================================================================

def richardson_lucy(image, psf, iterations):
    image = image.astype(np.float64)
    psf = psf.astype(np.float64)

    # Initialize the estimate with the blurred image
    estimate = np.full(image.shape, 0.5)

    # Flip the PSF for convolution
    psf_mirror = psf[::-1, ::-1]

    for _ in range(iterations):
        # Convolve estimate with PSF
        estimate_conv = convolve2d(estimate, psf, 'same', "symm")
        
        # Avoid division by zero
        estimate_conv[estimate_conv == 0] = 1e-10
        
        # Calculate relative blur
        relative_blur = image / estimate_conv
        
        # Update estimate
        estimate *= convolve2d(relative_blur, psf_mirror, 'same', "symm")
    
    return estimate

#================================================================================================================================================================================

def estimate_PSF(image, steps = 100, population = 1000, RL_iterations = 30, ksize = 15):

    if ( len(image.shape) == 3 ):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # initialize
    score = brenner(image)

    best_PSF = np.zeros((ksize, ksize))
    best_PSF[7,7] = 1.0

    T = 2.00
    for i in tqdm( range (0, steps) ):

        mutations = np.random.random( (population, ksize, ksize) ).astype(np.double) * T - ( T/2 )

        for k in tqdm( range (0, population) ):
            mutant = best_PSF + mutations[k,:,:]
            mutant = mutant / np.sum(mutant)

            estima = richardson_lucy(image, mutant, RL_iterations)
            if ( score < brenner(estima) ):
                best_PSF = mutant

        T = T * 0.98

    return best_PSF

#================================================================================================================================================================================    

def test_Richardson_Lucy():
    # Load the blurred image (replace 'blurred_image.png' with your image file)
    image = cv2.imread(filedialog.askopenfilename(), cv2.IMREAD_GRAYSCALE)

    # Define the PSF (example: a Gaussian PSF)
    # For a real-world application, replace this with your actual PSF
    psf = np.array(
        [
            [0, 1 , 2 , 4 , 8 , 4 , 2 , 1 , 0],
            [1, 2 , 4 , 8 , 16, 8 , 4 , 2 , 1],
            [2, 4 , 8 , 16, 32, 16, 8 , 4 , 2],
            [4, 8 , 16, 32, 64, 32, 16, 8 , 4],
            [8, 16, 32, 64, 64, 64, 32, 16, 8],
            [4, 8 , 16, 32, 64, 32, 16, 8 , 4],
            [2, 4 , 8 , 16, 32, 16, 8 , 4 , 2],
            [1, 2 , 4 , 8 , 16, 8 , 4 , 2 , 1],
            [0, 1 , 2 , 4 , 8 , 4 , 2 , 1 , 0],
        ]
    )

    psf = psf * np.random.random(psf.shape)
    psf = psf / np.sum(psf)

    blurred_image = convolve2d( image, psf[::-1, ::-1], 'same', "symm" ).astype(np.uint8)

    # Perform Richardson-Lucy deconvolution
    num_iterations = 30
    deblurred_image = richardson_lucy(blurred_image, psf, num_iterations)

    print( f" Original Image Brenner : {np.log( brenner(image) )}" )
    print( f" Blurred Image Brenner : {np.log( brenner(blurred_image) )}" )
    print( f" Deblurred Image Brenner : {np.log( brenner(deblurred_image) )}" )

    # Display the images
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    ax[0][0].imshow(image, cmap='gray', vmin=0, vmax=255)
    ax[0][0].set_title('Original Image')
    ax[0][0].axis('off')

    ax[0][1].imshow(psf, cmap='viridis')
    ax[0][1].set_title('PSF')
    ax[0][1].axis('off')

    ax[1][0].imshow(blurred_image, cmap='gray', vmin=0, vmax=255)
    ax[1][0].set_title('Blurred Image')
    ax[1][0].axis('off')

    ax[1][1].imshow(deblurred_image, cmap='gray', vmin=0, vmax=255)
    ax[1][1].set_title('Deblurred Image')
    ax[1][1].axis('off')

    plt.show()

    # Display the images
    vmin = min( np.min(image.astype(int) - deblurred_image.astype(int)), np.min(image.astype(int) - blurred_image.astype(int)) )
    vmax = max( np.max(image.astype(int) - deblurred_image.astype(int)), np.max(image.astype(int) - blurred_image.astype(int)) )

    fig, ax = plt.subplots(1, 2, figsize=(12, 10))
    ax[0].imshow((image.astype(int) - deblurred_image.astype(int)), cmap='coolwarm', vmin= vmin, vmax= vmax)
    ax[0].set_title('Difference map (original - deblurred)')
    ax[0].axis('off')

    ax[1].imshow((image.astype(int) - blurred_image.astype(int)), cmap='coolwarm', vmin= vmin, vmax= vmax)
    ax[1].set_title('Difference map (original - blurred)')
    ax[1].axis('off')

    plt.show()
