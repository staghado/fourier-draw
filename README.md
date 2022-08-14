# fourier-draw

Implementation of a technique allowing to draw the contour of any image using the Fourier transform.

# How to use it ?

usage: 
\
- **fourier-draw.py [-h] -i INPUT -o OUTPUT [-f FRAMES] [-N N]**

arguments:
```console
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        input image path
  -o OUTPUT, --output OUTPUT
                        output animation path
  -f FRAMES, --frames FRAMES
                        number of frames
  -N N, --N N           number of coefficients used
```

# Example

* Input image : 

<img src="images/raccoon_rock_n_roll.jpg" width="400" height="400" />

* Output animation :

<img src="animations/raccoon_epicycles_otsu.gif" width="800" height="800" />


# Improvements :

After the extraction of the contours, I only take the largest one for simplicity. But if we wanted to reproduce more fine-grained details
one would need to use all the available contours. In order to do that, we could simply stack all the contours into one big contour but that does not work pretty well since the naive stacking of contours adds more lines which make the final animation cumbersome.

We would like to connect the contours so as to minimize the intersection between the lines.

# Reference : 
- https://www.youtube.com/watch?v=r6sGWTCMz2k : But what is a Fourier series? From thermal transfer to designs with circles
