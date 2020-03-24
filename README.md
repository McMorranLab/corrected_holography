# corrected_holography

This repository contains python functions for creating arbitrary off-axis corrected hologram patterns. It accompanies the research article:

C. W. Johnson, J. S. Pierce, R. C. Moraski, A. E. Turner, A. T. Greenberg, W. S. Parker, and B. J. McMorran, “Corrected Off-Axis Holography for Producing Arbitrary Scalar Fields,” Opt. Express, OE (Submitted 2020)

It is managed by the McMorran Lab at the University of Oregon. See licence for use and distribution permissions.

POC: [Benjamin J. McMorran](mailto:mcmorran@uoregon.edu?subject=[GitHub]%20corrected_holography)

## Purpose

This repository is a Python implementation of the inversion of Eq. (13) as described in Section 2.3 of the article:

![equation](https://latex.codecogs.com/gif.latex?A_1Z_1%28%5Cmathbf%7Br%7D%29%5CTheta_1%28%5Cmathbf%7Br%7D%29%26%20%3D%20e%5E%7Bi%5Ctilde%7B%5Ceta%7D%28d-c_0%28%5Cmathbf%7Br%7D%29hZ%28%5Cmathbf%7Br%7D%29%29%7D%5CTheta%28%5Cmathbf%7Br%7D%29%5Csum_%7Bs%5Cin%20E_1%7D%5Cprod_%7Bn%3D1%7D%5E%5Cinfty%5Calpha_n%28%5Cmathbf%7Br%7D%29%5E%7Bs%28n%29%7DI_%7Bs%28n%29%7D%5Cbig%28-2i%5Ctilde%7B%5Ceta%7D%7Cc_n%28%5Cmathbf%7Br%7D%29%7ChZ%28%5Cmathbf%7Br%7D%29%5Cbig%29%20%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%2813%29)


It contains functions to search for the maps ![equation](https://latex.codecogs.com/gif.latex?s%28n%29%20%5Cin%20E_m) which contribute most to the sum on the RHS, and to numerically invert the above equation using those maps to find the functions ![equation](https://latex.codecogs.com/gif.latex?Z%28%5Cmathbf%7Br%7D%29) and ![equation](https://latex.codecogs.com/gif.latex?%5CTheta%28%5Cmathbf%7Br%7D%29) corresponding to a desired function ![equation](https://latex.codecogs.com/gif.latex?Z_1%28%5Cmathbf%7Br%7D%29%5CTheta_1%28%5Cmathbf%7Br%7D%29).

## Usage

`corrected_holography_example.py` contains a general working example of how to use the functions defined in `corrected_holography.py`. It is written as a notebook, and can be run using Hydrogen, or imported as a Jupyter notebook. Following a clone of this repository, it can be run immediately, with the following Python module dependencies: `copy`, `gc`, `heapq`, `itertools`, `matplotlib`, `numpy`, `os`, `scipy`, `time`.

In addition to illustrating the process described in the paper, this repository is intended to be a utility for those hoping to generate off-axis holograms for wavefront shaping. Thus, a bare-bones template `ch_template.py` is included that can be edited in a text editor but run from the command line. It has the extra dependency `argparse`.

In general, the user will need to specify the following variables in order to generate an off-axis hologram:

|Symbol|Variable name in `ch_template.py`|Description|
|-------|------|---|
|![equation](https://latex.codecogs.com/gif.latex?%5Ctilde%5Ceta)|`eta`|Complex parameter the attenuation and phase shift per unit length of material.|
|![equation](https://latex.codecogs.com/gif.latex?h)|`h`|The groove depth.|
|![equation](https://latex.codecogs.com/gif.latex?p)|`p`|Search limit parameter ![equation](https://latex.codecogs.com/gif.latex?%7Cs%28n%29%7C%5Cleq%20p). |
|![equation](https://latex.codecogs.com/gif.latex?q)|`q`|Search limit parameter ![equation](https://latex.codecogs.com/gif.latex?n%5Cleq%20q). |
|![equation](https://latex.codecogs.com/gif.latex?m)|`ord`|Diffraction order on which to imprint the desired function.|
|![equation](https://latex.codecogs.com/gif.latex?Z_1%28%5Cmathbf%7Br%7D%29%5CTheta_1%28%5Cmathbf%7Br%7D%29)|`beam_func`|A python function which defines the desired transverse profile.|
||`limit_search`|Boolean to limit the search to 1 non-zero value in ![eqaution](https://latex.codecogs.com/gif.latex?s%28n%29). Use only for visualization. |
||`num`|The output hologram will have resolution `num`x`num`. |
||`size`|The diameter in ![equation](https://latex.codecogs.com/gif.latex?%5Cmu%20m) of the grating. |
||`pitch`|The pitch in ![equation](https://latex.codecogs.com/gif.latex?%5Cmu%20m) of the grating. |
||`f_comps`|Required only for `arbitrary`. A python function which defines the Fourier components of the groove profile.|

### `ch_template.py` usage

`ch_template.py` is a command-line interface for `corrected_holography.py`. It will save a `.npy` file containing the computed hologram, as well as a `.npy` file containing the computed back Fourier plane, if specified.

The user must specify:
1. The desired beam profile within `ch_template.py`.
2. The desired groove profile (`sinusoidal`, `binary`, `blazed`, or `arbitrary`) via command line.
     * If `arbitrary`, the user must specify a groove profile within `ch_template.py`

All other grating parameters can be specified via the command line, and default to those listed in the article.

#### Example

To create a binary hologram that creates a Laguerre-Gauss mode LG<sub>2</sub><sup>3</sup> in the first diffracted order, with pitch 4um, the user would first define the LG mode in `ch_template.py`.
```
...
### Desired beam, desired groove profile
beam_func = LG23(x,y)

def LG23(x, y): ### user input
     ...
...
```
then run from the command line
```
python ch_template.py -fpath filepath -fname filename -pitch 4 -bfp binary
```
`-bfp` tells the program to compute and save the back fourier plane of the computed hologram. Also note that `corrected_holography.py` contains definitions of the Laguerre-Gauss modes for convenience.

The user can also specify an arbitrary groove profile by defining `f_comps`:

```
### Desired beam, desired groove profile
beam_func = ### user input - beam profile
f_comps = ### user input - groove profile's Fourier components
```

 and would then instead run
```
python ch_template.py -fpath filepath -fname filename -pitch 4 -bfp arbitrary
```
