
"""
ch_template.py is a command-line implementation of corrected off-axis
hologram pattern generation.

Copyright (C) 2020, Cameron W. Johnson and Jordan S. Pierce

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

# Required external dependencies to run ch_template.py.

# %load_ext autoreload
# %autoreload
import numpy
import matplotlib.pyplot as pyplot
import corrected_holography as ch
import os
import argparse

################################################################################
##################### Options - see README.md ##################################
################################################################################

### Desired beam, desired groove profile
beam_func = lambda x, y: ch.LG(x, y, 5, 3, 0.1*size)
f_comps = ch.cbla

### Default parameters
eta = 0.008j - numpy.pi / 29
h = 60
p = 5
q = 6
ord = 1
limit_search = False
num = 512
size = 30.
pitch = .866

################################################################################
################################################################################
################################################################################

grid = numpy.ogrid[size/2:-size/2:num*1j, -size/2:size/2:num*1j]
y, x = numpy.broadcast_arrays(*grid)
beam = beam_func(x, y)

def arbGrat(eta, h, p, q, ord, limit_search, num, size, pitch, beam_func, f_comps):
    """
    Creates an off-axis hologram for an arbitrary groove profile, defined by its fourier components.
    """
    # Choose a path name before running
    fpath = 'matches_sorted'
    fname = 'matches_groove({})_h({})_p({})_q({})_limit({})'.format(f_comps.__name__,h,p,q,limit_search)

    if os.path.exists('{}/{}.npy'.format(fpath,fname)):
        matches_sorted = numpy.load('{}/{}.npy'.format(fpath,fname),encoding='bytes',allow_pickle=True).item()
    else:
        matches_sorted = ch.first_order_contributions(f_comps, eta, h, p, q, ord, fpath, fname, limit_search)

    x_curve = numpy.ogrid[h-2:h+2:100j]    # the peak should be within these values
    response = ch.curve_arbitrary(x_curve,f_comps,eta,matches_sorted)
    Z_curve, A_curve = numpy.abs(response), numpy.unwrap(numpy.angle(response))
    max_inv = numpy.where(numpy.diff(Z_curve) < 0)[0][0]
    max_inv = x_curve[max_inv]
    print("The maximum invertible depth (computed profile) is {:.2f} nm".format(max_inv))

    x_curve = numpy.ogrid[1e-9:max_inv:1000j]
    response = ch.curve_arbitrary(x_curve,f_comps,eta,matches_sorted)
    Z_curve, A_curve = numpy.abs(response), numpy.unwrap(numpy.angle(response))

    # calculate grating
    grating = ch.correct_grating_arbitrary(beam, max_inv, x, Z_curve, A_curve, x_curve, f_comps, 199, pitch)

    return(grating, max_inv)

def sinGrat(eta, h, p, q, ord, limit_search, num, size, pitch, beam_func, f_comps):
    """
    Creates a sinusoidal off-axis hologram.
    """
    sin_x_curve = numpy.ogrid[33:39:100j]    # the peak should be within these values
    sin_response = ch.curve_sinusoidal(sin_x_curve,eta)
    sin_Z_curve, sin_A_curve = numpy.abs(sin_response), numpy.unwrap(numpy.angle(sin_response))
    sin_max_inv = numpy.where(numpy.diff(sin_Z_curve) < 0)[0][0]
    sin_max_inv = sin_x_curve[sin_max_inv]
    print("The maximum invertible depth (sinusoidal profile) is {:.2f} nm".format(sin_max_inv))

    sin_x_curve = numpy.ogrid[1e-9:sin_max_inv:1000j]
    sin_response = ch.curve_sinusoidal(sin_x_curve,eta)
    sin_Z_curve, sin_A_curve = numpy.abs(sin_response), numpy.unwrap(numpy.angle(sin_response))

    sin_grating = ch.correct_grating_sinusoidal(beam, sin_max_inv, x, sin_Z_curve, sin_A_curve, sin_x_curve, pitch)
    return(sin_grating, sin_max_inv)

def binGrat(eta, h, p, q, ord, limit_search, num, size, pitch, beam_func, f_comps):
    """
    Creates a binary off-axis hologram.
    """
    bin_max_inv = numpy.abs(numpy.pi/numpy.real(eta))
    print("The maximum invertible depth (binary profile) is {:.2f} nm".format(bin_max_inv))
    bin_grating = ch.correct_grating_binary(beam,size,199,pitch,sc=1.00)
    return(bin_grating, bin_max_inv)

def blaGrat(eta, h, p, q, ord, limit_search, num, size, pitch, beam_func, f_comps):
    """
    Creates a blazed off-axis hologram.
    """
    return(arbGrat(eta, h, p, q, ord, limit_search, num, size, pitch, beam_func, f_comps))

def bfp(grating, max_inv):
    """
    Calculates the back fourier plane of the grating.
    """
    # pad grating
    px = 2**13
    px2 = 2**12
    grat = numpy.zeros((px,px))
    grat[px2-num//2:px2+num//2,px2-num//2:px2+num//2] = grating
    # calculate back focal plane of each grating
    psi = numpy.exp(1.0j*grat*eta*max_inv)
    Psi = ch.fft2(psi)
    return(Psi)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-fpath', type = str, default = 'new_hologram', help = "Path to store the computed grating. ")
    parser.add_argument('-fname', type = str, default = None, help = "File name for the computed grating. ")
    parser.add_argument('-bfp', default = False, action = 'store_true', help = "Whether to calculate the back fourier plane of the computed grating. ")
    parser.add_argument('-eta', type = float, nargs = 2, default = (numpy.real(eta), numpy.imag(eta)), help = "Phase shift and attenuation parameter per nm (-eta Re[eta] Im[eta]). ")
    parser.add_argument('-h0', type = float, default = h, help = "Maximum groove depth. ")
    parser.add_argument('-p', type = int, default = p, help = "Search parameter. ")
    parser.add_argument('-q', type = int, default = q, help = "Search parameter. ")
    parser.add_argument('-ord', type = int, default = ord, help = "Diffraction order of desired profile. ")
    parser.add_argument('-limit_search', default = limit_search, action = 'store_true', help = "Limits search. Use for testing and visualization only.")
    parser.add_argument('-num', type = int, default = num, help = "Resolution of the output grating. ")
    parser.add_argument('-size', type = float, default = size, help = "Diameter of the grating in um. ")
    parser.add_argument('-pitch', type = float, default = pitch, help = "Pitch of the grating in um. ")

    sp = parser.add_subparsers(title = 'Groove profile. ', dest = 'Groove profile.', help = 'Groove profile. ')
    sp.required = True

    blaP = sp.add_parser("blazed")
    blaP.set_defaults(func=blaGrat, which = "blazed")

    binP = sp.add_parser("binary")
    binP.set_defaults(func=binGrat, which = "binary")

    sinP = sp.add_parser("sinusoidal")
    sinP.set_defaults(func=sinGrat, which = "sinusoidal")

    arbP = sp.add_parser("arbitrary")
    arbP.set_defaults(func=arbGrat, which = "arbitrary")

    args = parser.parse_args()
    grating, max_inv = args.func(args.eta[0]+1j*args.eta[1], args.h0, args.p, args.q, args.ord, args.limit_search, args.num, args.size, args.pitch, beam_func, f_comps)

    if args.fname is None:
        args.fname = "groove({})_eta({:.3f})({:.3f})_h0({})_p({})_q({})_ord({})_lim({})".format(
            args.which, args.eta[0],args.eta[1], args.h0, args.p, args.q, args.ord, args.limit_search)
    bfpname = args.fname + "_bfp"

    if os.path.exists(args.fpath):
        pass
    else:
        os.mkdir(args.fpath)

    numpy.save("{}/{}.npy".format(args.fpath, args.fname), grating)
    print("Computed grating has been saved to '{}/{}.npy'".format(args.fpath, args.fname))

    if args.bfp:
        bfplane = bfp(grating, max_inv)
        numpy.save("{}/{}.npy".format(args.fpath, bfpname), bfplane)
        print("Back fourier plane has been saved to '{}/{}.npy'".format(args.fpath, bfpname))
