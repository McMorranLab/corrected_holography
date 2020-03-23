# corrected_holography
This repository contains python functions for creating arbitrary off-axis corrected hologram patterns. It accompanies **[Ref the paper]**

## Purpose

This repository is a Python implementation of the inversion described in Section 2.3 of the paper:

![equation](https://latex.codecogs.com/gif.latex?A_1Z_1%28%5Cmathbf%7Br%7D%29%5CTheta_1%28%5Cmathbf%7Br%7D%29%26%20%3D%20e%5E%7Bi%5Ctilde%7B%5Ceta%7D%28d-c_0%28%5Cmathbf%7Br%7D%29hZ%28%5Cmathbf%7Br%7D%29%29%7D%5CTheta%28%5Cmathbf%7Br%7D%29%5Csum_%7Bs%5Cin%20E_1%7D%5Cprod_%7Bn%3D1%7D%5E%5Cinfty%5Calpha_n%28%5Cmathbf%7Br%7D%29%5E%7Bs%28n%29%7DI_%7Bs%28n%29%7D%5Cbig%28-2i%5Ctilde%7B%5Ceta%7D%7Cc_n%28%5Cmathbf%7Br%7D%29%7ChZ%28%5Cmathbf%7Br%7D%29%5Cbig%29)

It contains functions to search for the maps 
![equation](https://latex.codecogs.com/gif.latex?s%28n%29%20%5Cin%20E_m) 
which contribute most to the RHS, and to numerically invert the above equation using those maps to find the function 
![equation](https://latex.codecogs.com/gif.latex?Z%28%5Cmathbf%7Br%7D%29)
corresponding to a desired function 
![equation](https://latex.codecogs.com/gif.latex?Z_m%28%5Cmathbf%7Br%7D%29)
.

## Usage

`corrected_holography_example.py` contains a general working example of how to use the functions defined in `corrected_holography.py`. It is written as a notebook, and can be run using Hydrogen, or imported as a Jupyter notebook. Following a clone of this repository, it can be run immediately, with the following Python module dependencies: `copy`, `gc`, `heapq`, `itertools`, `matplotlib`, `numpy`, `os`, `scipy`, `time`.

In addition to illustrating the process described in the paper, this repository is intended to be a utility for those hoping to generate off-axis holograms for wavefront shaping. Thus, a bare-bones template `ch_template.py` is included that can be edited in a text editor but run from the command line. It has the extra dependency `argparse`.

In general, the user will need to specify the following variables in order to generate an off-axis hologram:

|Symbol|Variable name in `ch_template.py`|Description|
|-------|------|---|
|$\tilde{\eta}$|`eta`|Specifies the attenuation and phase shift of the material.|
|$h$|`h`|The groove depth.|
|$p$|`p`|Search parameter $\|s(n)\|\leq p$ |
|$q$|`q`|Search parameter $n\leq q$ |
|$m$|`ord`|Diffraction order on which to imprint the desired function.|
|$\Theta_m(\mathbf{r})Z_m(\mathbf{r})$|`beam_func`|A python function which defines the desired transverse profile.|
||`limit_search`|Boolean to limit the search to 1 non-zero value in $s(n)$. Use only for visualization. |
||`num`|The output hologram will have resolution `num`$\times$`num`. |
||`size`|The diameter in $\mu m$ of the grating. |
||`pitch`|The pitch in $\mu m$ of the grating. |
||`f_comps`|Required only for `arbitrary`. A python function which defines the Fourier components of the groove profile.|

To use `ch_template.py`, modify the variables at the top of the code, then run via the command line. The output will be a `.npy` file containing the grating values, along with a `.npy` containing the far-field diffraction pattern if `-bfp` is used. 

## Funding

Necessary?

## Acknowledgments
Same as paper?
