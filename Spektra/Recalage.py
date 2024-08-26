import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tkinter import filedialog as fd

#======================================================================================================================================================================================================================================================

def Maxima_loco(img, bsize = 3, ksize = 3):

	if len(img.shape) == 3 :
		img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	img = cv.GaussianBlur(img, (bsize,bsize), 0)


	# Apply Sobelx in high output datatype 'float32'
	# and then converting back to 8-bit to prevent overflow
	sobelx_64 = cv.Sobel(img,cv.CV_32F,1,0,ksize=ksize)
	absx_64 = np.absolute(sobelx_64)
	sobelx_8u1 = absx_64/absx_64.max()*255
	sobelx_8u = np.uint8(sobelx_8u1)
	 
	# Similarly for Sobely
	sobely_64 = cv.Sobel(img,cv.CV_32F,0,1,ksize=ksize)
	absy_64 = np.absolute(sobely_64)
	sobely_8u1 = absy_64/absy_64.max()*255
	sobely_8u = np.uint8(sobely_8u1)
	 
	# From gradients calculate the magnitude and changing
	# it to 8-bit (Optional)
	mag = np.hypot(sobelx_8u, sobely_8u)
	mag = mag/mag.max()*255
	mag = np.uint8(mag)
	 
	# Find the direction and change it to degree
	theta = np.arctan2(sobely_64, sobelx_64)
	angle = np.rad2deg(theta)

	# Find the neighbouring pixels (b,c) in the rounded gradient direction
	# and then apply non-max suppression
	M, N = mag.shape
	Non_max = np.zeros((M,N), dtype= np.uint8)

	for i in range(1,M-1):
		for j in range(1,N-1):
		   # Horizontal 0
			if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180) or (-22.5 <= angle[i,j] < 0) or (-180 <= angle[i,j] < -157.5):
				b = mag[i, j+1]
				c = mag[i, j-1]
				
			# Diagonal 45
			elif (22.5 <= angle[i,j] < 67.5) or (-157.5 <= angle[i,j] < -112.5):
				b = mag[i+1, j+1]
				c = mag[i-1, j-1]
				
			# Vertical 90
			elif (67.5 <= angle[i,j] < 112.5) or (-112.5 <= angle[i,j] < -67.5):
				b = mag[i+1, j]
				c = mag[i-1, j]
				
			# Diagonal 135
			elif (112.5 <= angle[i,j] < 157.5) or (-67.5 <= angle[i,j] < -22.5):
				b = mag[i+1, j-1]
				c = mag[i-1, j+1]	
					   
				
			# Non-max Suppression
			if (mag[i,j] >= b) and (mag[i,j] >= c):
				Non_max[i,j] = mag[i,j]
				
			else:
				Non_max[i,j] = 0

	return Non_max

#======================================================================================================================================================================================================================================================

def Hysteresis_8x(img, sb = 50, sh = 150, bsize = 3, ksize = 3):

	if len(img.shape) == 3 :
		img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	img = cv.GaussianBlur(img, (bsize,bsize), 0)


	# Apply Sobelx in high output datatype 'float32'
	# and then converting back to 8-bit to prevent overflow
	sobelx_64 = cv.Sobel(img,cv.CV_32F,1,0,ksize=ksize)
	absx_64 = np.absolute(sobelx_64)
	sobelx_8u1 = absx_64/absx_64.max()*255
	sobelx_8u = np.uint8(sobelx_8u1)
	 
	# Similarly for Sobely
	sobely_64 = cv.Sobel(img,cv.CV_32F,0,1,ksize=ksize)
	absy_64 = np.absolute(sobely_64)
	sobely_8u1 = absy_64/absy_64.max()*255
	sobely_8u = np.uint8(sobely_8u1)
	 
	# From gradients calculate the magnitude and changing
	# it to 8-bit (Optional)
	mag = np.hypot(sobelx_8u, sobely_8u)
	mag = mag/mag.max()*255
	mag = np.uint8(mag)
	 
	# Find the direction and change it to degree
	theta = np.arctan2(sobely_64, sobelx_64)
	angle = np.rad2deg(theta)

	# Find the neighbouring pixels (b,c) in the rounded gradient direction
	# and then apply non-max suppression
	M, N = mag.shape
	Non_max = np.zeros((M,N), dtype= np.uint8)

	for i in range(1,M-1):
		for j in range(1,N-1):
		   # Horizontal 0
			if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180) or (-22.5 <= angle[i,j] < 0) or (-180 <= angle[i,j] < -157.5):
				b = mag[i, j+1]
				c = mag[i, j-1]
				
			# Diagonal 45
			elif (22.5 <= angle[i,j] < 67.5) or (-157.5 <= angle[i,j] < -112.5):
				b = mag[i+1, j+1]
				c = mag[i-1, j-1]
				
			# Vertical 90
			elif (67.5 <= angle[i,j] < 112.5) or (-112.5 <= angle[i,j] < -67.5):
				b = mag[i+1, j]
				c = mag[i-1, j]
				
			# Diagonal 135
			elif (112.5 <= angle[i,j] < 157.5) or (-67.5 <= angle[i,j] < -22.5):
				b = mag[i+1, j-1]
				c = mag[i-1, j+1]		   
				
			# Non-max Suppression
			if (mag[i,j] >= b) and (mag[i,j] >= c):
				Non_max[i,j] = mag[i,j]
			else:
				Non_max[i,j] = 0

	# Set high and low threshold
	highThreshold = sh
	lowThreshold = sb

	M, N = Non_max.shape
	out = np.zeros((M,N), dtype= np.uint8)

	# If edge intensity is greater than 'High' it is a sure-edge
	# below 'low' threshold, it is a sure non-edge
	strong_i, strong_j = np.where(Non_max >= highThreshold)
	zeros_i, zeros_j = np.where(Non_max < lowThreshold)

	# weak edges
	weak_i, weak_j = np.where((Non_max <= highThreshold) & (Non_max >= lowThreshold))

	# Set same intensity value for all edge pixels
	out[strong_i, strong_j] = 255
	out[zeros_i, zeros_j ] = 0
	out[weak_i, weak_j] = 75

	M, N = out.shape
	for i in range(1, M-1):
		for j in range(1, N-1):
			if (out[i,j] == 75):
				if 255 in [out[i+1, j-1],out[i+1, j],out[i+1, j+1],out[i, j-1],out[i, j+1],out[i-1, j-1],out[i-1, j],out[i-1, j+1]]:
					out[i, j] = 255
				else:
					out[i, j] = 0

	return out

#======================================================================================================================================================================================================================================================

def Hysteresis_8x_v2(img, sb = 50, sh = 150, bsize = 3, ksize = 3):

	if len(img.shape) == 3 :
		img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	img = cv.GaussianBlur(img, (bsize,bsize), 0)


	# Apply Sobelx in high output datatype 'float32'
	# and then converting back to 8-bit to prevent overflow
	sobelx_64 = cv.Sobel(img,cv.CV_32F,1,0,ksize=ksize)
	absx_64 = np.absolute(sobelx_64)
	sobelx_8u1 = absx_64/absx_64.max()*255
	sobelx_8u = np.uint8(sobelx_8u1)
	 
	# Similarly for Sobely
	sobely_64 = cv.Sobel(img,cv.CV_32F,0,1,ksize=ksize)
	absy_64 = np.absolute(sobely_64)
	sobely_8u1 = absy_64/absy_64.max()*255
	sobely_8u = np.uint8(sobely_8u1)
	 
	# From gradients calculate the magnitude and changing
	# it to 8-bit (Optional)
	mag = np.hypot(sobelx_8u, sobely_8u)
	mag = mag/mag.max()*255
	mag = np.uint8(mag)

	# Set high and low threshold
	highThreshold = sh
	lowThreshold = sb

	M, N = img.shape
	out = np.zeros((M,N), dtype= np.uint8)

	# If edge intensity is greater than 'High' it is a sure-edge
	# below 'low' threshold, it is a sure non-edge
	strong_i, strong_j = np.where(mag >= highThreshold)
	zeros_i, zeros_j = np.where(mag < lowThreshold)

	# weak edges
	weak_i, weak_j = np.where((mag <= highThreshold) & (mag >= lowThreshold))

	# Set same intensity value for all edge pixels
	out[strong_i, strong_j] = 255
	out[zeros_i, zeros_j ] = 0
	out[weak_i, weak_j] = 75

	M, N = out.shape
	for i in range(1, M-1):
		for j in range(1, N-1):
			if (out[i,j] == 75):
				if 255 in [out[i+1, j-1],out[i+1, j],out[i+1, j+1],out[i, j-1],out[i, j+1],out[i-1, j-1],out[i-1, j],out[i-1, j+1]]:
					out[i, j] = 255
				else:
					out[i, j] = 0

	return out

#======================================================================================================================================================================================================================================================

def Hysteresis_4x(img, sb = 50, sh = 150, bsize = 3, ksize = 3):

	if len(img.shape) == 3 :
		img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	img = cv.GaussianBlur(img, (bsize,bsize), 0)

	# Apply Sobelx in high output datatype 'float32'
	# and then converting back to 8-bit to prevent overflow
	filter_x = np.matrix([[0,0,0],[-1,0,1],[0,0,0]], np.int8)
	sobelx_64 = cv.filter2D(img, cv.CV_32F, filter_x)
	absx_64 = np.absolute(sobelx_64)
	sobelx_8u1 = absx_64/absx_64.max()*255
	sobelx_8u = np.uint8(sobelx_8u1)
	 
	# Similarly for Sobely
	filter_y = np.matrix([[0,-1,0],[0,0,0],[0,1,0]], np.int8)
	sobely_64 = cv.filter2D(img, cv.CV_32F, filter_y)
	absy_64 = np.absolute(sobely_64)
	sobely_8u1 = absy_64/absy_64.max()*255
	sobely_8u = np.uint8(sobely_8u1)
	 
	# From gradients calculate the magnitude and changing
	# it to 8-bit (Optional)
	mag = np.hypot(sobelx_8u, sobely_8u)
	mag = mag/mag.max()*255
	mag = np.uint8(mag)
	 
	# Find the direction and change it to degree
	theta = np.arctan2(sobely_64, sobelx_64)
	angle = np.rad2deg(theta)

	# Find the neighbouring pixels (b,c) in the rounded gradient direction
	# and then apply non-max suppression
	M, N = mag.shape
	Non_max = np.zeros((M,N), dtype= np.uint8)

	for i in range(1,M-1):
		for j in range(1,N-1):
		   # Horizontal 0
			if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180) or (-22.5 <= angle[i,j] < 0) or (-180 <= angle[i,j] < -157.5):
				b = mag[i, j+1]
				c = mag[i, j-1]
			# Diagonal 45
			elif (22.5 <= angle[i,j] < 67.5) or (-157.5 <= angle[i,j] < -112.5):
				b = mag[i+1, j+1]
				c = mag[i-1, j-1]
			# Vertical 90
			elif (67.5 <= angle[i,j] < 112.5) or (-112.5 <= angle[i,j] < -67.5):
				b = mag[i+1, j]
				c = mag[i-1, j]
			# Diagonal 135
			elif (112.5 <= angle[i,j] < 157.5) or (-67.5 <= angle[i,j] < -22.5):
				b = mag[i+1, j-1]
				c = mag[i-1, j+1]		   
				
			# Non-max Suppression
			if (mag[i,j] >= b) and (mag[i,j] >= c):
				Non_max[i,j] = mag[i,j]
			else:
				Non_max[i,j] = 0

	# Set high and low threshold
	highThreshold = sh
	lowThreshold = sb

	M, N = Non_max.shape
	out = np.zeros((M,N), dtype= np.uint8)

	# If edge intensity is greater than 'High' it is a sure-edge
	# below 'low' threshold, it is a sure non-edge
	strong_i, strong_j = np.where(Non_max >= highThreshold)
	zeros_i, zeros_j = np.where(Non_max < lowThreshold)

	# weak edges
	weak_i, weak_j = np.where((Non_max <= highThreshold) & (Non_max >= lowThreshold))

	# Set same intensity value for all edge pixels
	out[strong_i, strong_j] = 255
	out[zeros_i, zeros_j ] = 0
	out[weak_i, weak_j] = 75

	M, N = out.shape
	for i in range(1, M-1):
		for j in range(1, N-1):
			if (out[i,j] == 75):
				if 255 in [out[i+1, j-1],out[i+1, j],out[i+1, j+1],out[i, j-1],out[i, j+1],out[i-1, j-1],out[i-1, j],out[i-1, j+1]]:
					out[i, j] = 255
				else:
					out[i, j] = 0

	return out

#======================================================================================================================================================================================================================================================

def Hysteresis_4x_v2(img, sb = 50, sh = 150, bsize = 3, ksize = 3):

	if len(img.shape) == 3 :
		img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

	img = cv.GaussianBlur(img, (bsize,bsize), 0)

	# Apply Sobelx in high output datatype 'float32'
	# and then converting back to 8-bit to prevent overflow
	filter_x = np.matrix([[0,0,0],[-1,0,1],[0,0,0]], np.int8)
	sobelx_64 = cv.filter2D(img, cv.CV_32F, filter_x)
	absx_64 = np.absolute(sobelx_64)
	sobelx_8u1 = absx_64/absx_64.max()*255
	sobelx_8u = np.uint8(sobelx_8u1)
	 
	# Similarly for Sobely
	filter_y = np.matrix([[0,-1,0],[0,0,0],[0,1,0]], np.int8)
	sobely_64 = cv.filter2D(img, cv.CV_32F, filter_y)
	absy_64 = np.absolute(sobely_64)
	sobely_8u1 = absy_64/absy_64.max()*255
	sobely_8u = np.uint8(sobely_8u1)
	 
	# From gradients calculate the magnitude and changing
	# it to 8-bit (Optional)
	mag = np.hypot(sobelx_8u, sobely_8u)
	mag = mag/mag.max()*255
	mag = np.uint8(mag)

	# Find the neighbouring pixels (b,c) in the rounded gradient direction
	# and then apply non-max suppression
	M, N = mag.shape

	# Set high and low threshold
	highThreshold = sh
	lowThreshold = sb

	out = np.zeros((M,N), dtype= np.uint8)

	# If edge intensity is greater than 'High' it is a sure-edge
	# below 'low' threshold, it is a sure non-edge
	strong_i, strong_j = np.where(mag >= highThreshold)
	zeros_i, zeros_j = np.where(mag < lowThreshold)

	# weak edges
	weak_i, weak_j = np.where((mag <= highThreshold) & (mag >= lowThreshold))

	# Set same intensity value for all edge pixels
	out[strong_i, strong_j] = 255
	out[zeros_i, zeros_j ] = 0
	out[weak_i, weak_j] = 75

	M, N = out.shape
	for i in range(1, M-1):
		for j in range(1, N-1):
			if (out[i,j] == 75):
				if 255 in [out[i+1, j-1],out[i+1, j],out[i+1, j+1],out[i, j-1],out[i, j+1],out[i-1, j-1],out[i-1, j],out[i-1, j+1]]:
					out[i, j] = 255
				else:
					out[i, j] = 0

	return out

#======================================================================================================================================================================================================================================================

def Sobel(img, ksize = 3, scale = 1, delta = 0, ddepth = cv.CV_16S, borderType = cv.BORDER_DEFAULT):

	if len(img.shape) == 3 :
		img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	img = cv.GaussianBlur(img, (3,3), 0)

	Gx = cv.Sobel(img, ddepth, 1, 0, ksize = ksize, scale = scale, delta = delta, borderType = borderType)
	Gy = cv.Sobel(img, ddepth, 0, 1, ksize = ksize, scale = scale, delta = delta, borderType = borderType)

	Abs_Gx = cv.convertScaleAbs(Gx)
	Abs_Gy = cv.convertScaleAbs(Gy)

	Gr = cv.addWeighted(Abs_Gx, 0.5, Abs_Gy, 0.5, 0)
	return Gr

#======================================================================================================================================================================================================================================================

def Gradient_X(img):
	filter_x = np.matrix([[0,0,0],[-1,0,1],[0,0,0]], np.int8)

	Gx = cv.filter2D(img, cv.CV_32F, filter_x)
	Gx = np.absolute(Gx)
	Gx = Gx/Gx.max()*255
	Gx = np.uint8(Gx)

	return Gx

#======================================================================================================================================================================================================================================================

def Gradient_Y(img):
	filter_y = np.matrix([[0,-1,0],[0,0,0],[0,1,0]], np.int8)

	Gy = cv.filter2D(img, cv.CV_32F, filter_y)
	Gy = np.absolute(Gy)
	Gy = Gy/Gy.max()*255
	Gy = np.uint8(Gy)

	return Gy

#======================================================================================================================================================================================================================================================

def Sobel_4x(img):

	if len(img.shape) == 3 :
		img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	img = cv.GaussianBlur(img, (3,3), 0)

	Gx = Gradient_X(img)
	Gy = Gradient_Y(img)

	Abs_Gx = cv.convertScaleAbs(Gx)
	Abs_Gy = cv.convertScaleAbs(Gy)

	Gr = cv.addWeighted(Abs_Gx, 0.5, Abs_Gy, 0.5, 0)
	return Gr

#======================================================================================================================================================================================================================================================

def Sobel_8x(img):

	if len(img.shape) == 3 :
		img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	img = cv.GaussianBlur(img, (3,3), 0)

	filter_x = np.matrix([[-1,0,1],[-2,0,2],[-1,0,1]], np.int8)
	filter_y = np.matrix([[-1,-2,-1],[0,0,0],[1,2,1]], np.int8)

	Gx = cv.filter2D(img, cv.CV_32F, filter_x)
	Gx = np.absolute(Gx)
	Gx = Gx/Gx.max()*255
	Gx = np.uint8(Gx)

	Gy = cv.filter2D(img, cv.CV_32F, filter_y)
	Gy = np.absolute(Gy)
	Gy = Gy/Gy.max()*255
	Gy = np.uint8(Gy)

	Abs_Gx = cv.convertScaleAbs(Gx)
	Abs_Gy = cv.convertScaleAbs(Gy)

	Gr = cv.addWeighted(Abs_Gx, 0.5, Abs_Gy, 0.5, 0)
	return Gr

#======================================================================================================================================================================================================================================================

def Laplace(img, ksize = 3, ddepth = cv.CV_16S):

	if len(img.shape) == 3 :
		img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	img = cv.GaussianBlur(img, (3,3), 0)

	result = cv.Laplacian(img, ddepth, ksize = ksize)
	result = cv.convertScaleAbs(result)
	
	return result

#======================================================================================================================================================================================================================================================

def measure_LAP(img, ksize = 3, ddepth = cv.CV_64F):

	if len(img.shape) == 3 :
		img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	result = cv.Laplacian(img, ddepth, ksize = ksize)
	result = cv.convertScaleAbs(result)

	w, h = img.shape

	return np.sum(result)

#======================================================================================================================================================================================================================================================

def measure_LAP_VAR(img, ksize = 3, ddepth = cv.CV_64F):

	img 	= cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	result 	= cv.Laplacian(img, ddepth, ksize = ksize)
	result 	= cv.convertScaleAbs(result)

	L = np.mean(result)
	w, h = img.shape

	return np.sum( (result - L)**2 )

#======================================================================================================================================================================================================================================================

def brenner(img):
    '''
    :param img:narray             the clearer the image,the larger the return value
    :return: float 
    '''
    shape = np.shape(img)
    
    out = 0
    for y in range(0, shape[1]):
        for x in range(0, shape[0]-2):
            
            out+=(int(img[x+2,y])-int(img[x,y]))**2
            
    return out