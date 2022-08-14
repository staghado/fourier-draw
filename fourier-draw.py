#!/usr/bin/env python3
# coding: utf-8

### Fourier Transform drawings

import argparse
from math import tau

import cv2
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad_vec

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, type=str,
	help="input image path")
ap.add_argument("-o", "--output", required=True, type=str,
	help="output animation path")
ap.add_argument("-f", "--frames", required=False, default=300, type=int,
	help="number of frames")
ap.add_argument("-N", "--N", required=False, default=300, type=int,
	help="number of coefficients used")
args = vars(ap.parse_args())

assert args["N"] >= args["frames"], "the number of coefficients used needs to be higher than the number of frames"

# load an image
img = cv2.imread(args["input"])

# convert the image from RGB to grayscale & add Gaussian Blur to remove high frequency edges
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(imgray, (7, 7), 0)

# threshold the grayscale image to get a binary image
#ret, thresh = cv2.threshold(imgray, 235, 255, 0)

(T, thresh) = cv2.threshold(blurred, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

# extract the contours
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# get the indice of the largest contour in the image
largest_contour_idx = np.argmax([len(c) for c in contours])

# create a list of tuple coordinates of the contour points
verts = [ tuple(coord) for coord in contours[largest_contour_idx].squeeze() ]

# extract the xs and ys
xs, ys = zip(*verts)

# center the contour around (0,0)
xs = np.asarray(xs) - np.mean(xs)
ys = - np.asarray(ys) + np.mean(ys)

# create a parameter t_list from [0, tau] to parametrize the contour
t_list = np.linspace(0, tau, len(xs))


### Compute the Fourier coefficients

def f(t, t_list, xs, ys):
    """
    the interpolation(linear) of the contour points between xs and ys.

    Args:
        t (float): time t
        t_list (list): list of times in [0, tau]
        xs (array): x coords
        ys (array): y coords

    Returns:
        interpolation of the contour at the time t
    """
    return np.interp(t, t_list, xs + 1j*ys)


def compute_cn(f, n):
    """
    This function calcualtes the Fourier coefficients by numerical integration.

    Args:
        f (function): the contour interpolated function
        n (integer): Fourier indice of "c_n"

    Returns:
        coef (complex): the "c_n" coefficient
    """
    coef = 1/tau*quad_vec(
        lambda t: f(t, t_list, xs, ys)*np.exp(-n*t*1j), 
        0, 
        tau, 
        limit=100, # limit of convergence of the numerical integration method
        full_output=False)[0]
    return coef

def get_circle_coords(center, r, N=50):
    """
    Get the coordinates of the cirle of radius 'r' and center 'center'

    Args:
        center (tuple): the center
        r (float): the radius
        N (int, optional): number of points used to approximate teh circle. Defaults to 50.

    Returns:
        x, y: two lists of size N of coordinates laying on the circle.
    """
    theta = np.linspace(0, tau, N)
    x, y = center[0] + r * np.cos(theta), center[1] + r * np.sin(theta)
    return x, y

def get_next_pos(c, fr, t, drawing_time = 1):
    """
    Get the rotated vector 'c_n' at time t and at frequency fr.

    Args:
        c (complex): c_n 
        fr (_type_): n
        t (float): time t
        drawing_time (int, optional): total drawing time. Defaults to 1.

    Returns:
        rotated vector
    """
    angle = (fr * tau * t) / drawing_time
    return c * np.exp(1j*angle)


N = args["N"]
coefs = [ (compute_cn(f, 0), 0) ] + [ (compute_cn(f, j), j) for i in range(1, N+1) for j in (i, -i) ]

   
# creating a blank window for the animation 
fig, ax = plt.subplots()

circles = [ax.plot([], [], 'b-')[0] for i in range(-N, N+1)]
circle_lines = [ax.plot([], [], 'g-')[0] for i in range(-N, N+1)]
drawing, = ax.plot([], [], 'r-', linewidth=2)

# to fix the size of figure so that the figure does not get cropped/trimmed
ax.set_xlim(-500, 500)
ax.set_ylim(-500, 500)

ax.set_axis_off()
ax.set_aspect('equal')
fig.set_size_inches(15, 15)
   
# initializing empty values
# for x and y co-ordinates
draw_x, draw_y = [], []
   
# animation function 
def animate(i, coefs, time): 
    # t is a parameter which varies
    # with the frame number
    t = time[i]
    
    coefs = [ (get_next_pos(c, fr, t=t), fr) for c, fr in coefs ]
    center = (0, 0)
    for i, elts in enumerate(coefs) :
        c, _ = elts
        r = np.linalg.norm(c)
        x, y = get_circle_coords(center=center, r=r, N=80)
        circle_lines[i].set_data([center[0], center[0]+np.real(c)], [center[1], center[1]+np.imag(c)])
        circles[i].set_data(x, y) 
        center = (center[0] + np.real(c), center[1] + np.imag(c))
    
    # center points now are points from last circle
    # these points are used as drawing points
    draw_x.append(center[0])
    draw_y.append(center[1])

    # draw the curve from last point
    drawing.set_data(draw_x, draw_y)
   
# calling the animation function 
drawing_time = 1
frames = args["frames"]
time = np.linspace(0, drawing_time, num=frames)    
anim = animation.FuncAnimation(fig, animate, frames = frames, interval = 5, fargs=(coefs, time)) 

# saves the animation in our desktop
anim.save(args["output"], fps = 15)


