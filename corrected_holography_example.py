
"""
corrected_holography_example.py is an example implementation for corrected off-axis hologram pattern generation

Copyright (C) 2020, Cameron Johnson

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

Change the file path name in line 63 before running.
"""

# %% Code Cell ##################################################################

"""
Required external dependencies to run notebook. corrected_holography.py is the corrected holography definitions
python script containing all the special functions used below and is included in the same github repository.
"""
%load_ext autoreload
%autoreload
import numpy
import matplotlib.pyplot as pyplot
import corrected_holography as ch
import os

# %% Code Cell ##################################################################

"""
Creates dictionary of ranked maps s_j(n) of order=ord for a blazed grating. Takes about 30 minutes to run on my personal computer.
Saves file named "fname.npy" to file path "fpath". If the file "fpath/fname.npy" already exists, it will be loaded instead.

Returned dictionary format:
matches_sorted  = {
                    0{j=1}: { 'list':  [[1,s_1(1)],[2,s_1(2)],[3,s_1(3)],[4,s_1(4)],...],
                              'order': ord,
                              'val':   C_1 },
                    1:      { 'list':  [[1,s_2(1)],[2,s_2(2)],[3,s_2(3)],[4,s_2(4)],...],
                              'order': ord,
                              'val':   C_2 },
                    ...
                  }
"""

energy = 200                 # electron energy in keV
eta = 0.008j - numpy.pi / 29 # material parameter of Si3N4 for 200 keV electrons (attenuation and phase shift nm^-1)
h = 60                       # the search depth in nm (this should be around the maximum invertible depth
                             # (depth of maximum diffraction efficiency))
p = 5                        # maximum value of |s(n)| in brute force search
q = 6                        # maximum value of n for non-zero s(n) in brute force search
ord = 1                      # order of maps desired i.e. diffraction order in which desired function will appear

# Change the path name before running
fpath = '/Volumes/GoogleDrive/My Drive/Johnson_Research/cwj_projects/cwj_CorrectedHolography/cwj_ch_code/matches_sorted'
fname = 'matches_E({})eV_h({})_p({})_q({})'.format(energy,h,p,q)

if os.path.exists('{}/{}.npy'.format(fpath,fname)):
    matches_sorted = numpy.load('{}/{}.npy'.format(fpath,fname),encoding='bytes',allow_pickle=True).item()
else:
    matches_sorted = ch.first_order_contributions(ch.cbla, eta, h, p, q, ord, fpath, fname)

ch.print_sns_table(matches_sorted)


# %% Code Cell ##################################################################

"""
Plots the relative values of the ranked contributions C_j of the computed maps for the blazed groove profile
"""

vals = [numpy.log10(matches_sorted[i]['val']/matches_sorted[0]['val']) for i in range(50000)]

pyplot.figure(figsize=(5,5),dpi=300)
pyplot.plot(vals,label=r'$h_0$ = {} nm'.format(h),linewidth=1)
pyplot.title("Contributing values for the first 50000 terms")
pyplot.ylabel(r"$\log_{10}(C_j/C_0)$")
pyplot.xlabel(r"Rank $j$")
pyplot.grid(which='both',alpha=0.2)
pyplot.xticks([0,10000,20000,30000,40000,50000])
pyplot.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
pyplot.xlim([0,50000])
pyplot.show()

# %% Code Cell ##################################################################

"""
Finds the true maximum invertable mill depth (max_inv) (depth of maximum diffraction efficiency) of the groove
profile with the computed maps, as well as for sinusoidal and binary groove profiles.
"""

bla_x_curve = numpy.ogrid[58:62:100j]    # the peak should be within these values
bla_response = ch.curve_arbitrary(bla_x_curve,ch.cbla,eta,matches_sorted)
bla_Z_curve, A_curve = numpy.abs(bla_response), numpy.unwrap(numpy.angle(bla_response))
bla_max_inv = numpy.where(numpy.diff(bla_Z_curve) < 0)[0][0]
bla_max_inv = bla_x_curve[bla_max_inv]
print("The maximum invertible depth is {} nm".format(bla_max_inv))

sin_x_curve = numpy.ogrid[33:39:100j]    # the peak should be within these values
sin_response = ch.curve_sinusoidal(sin_x_curve,eta)
sin_Z_curve, sin_A_curve = numpy.abs(sin_response), numpy.unwrap(numpy.angle(sin_response))
sin_max_inv = numpy.where(numpy.diff(sin_Z_curve) < 0)[0][0]
sin_max_inv = sin_x_curve[sin_max_inv]
print("The maximum invertible depth is {} nm".format(sin_max_inv))

bin_max_inv = 29
print("The maximum invertible depth is {} nm".format(bin_max_inv))

# %% Code Cell ##################################################################

"""
Maps the 1d numerical values of Z_1 that correspond to a linear input of Z, this map is inverted  to interpolate
the 2d arrays Z(x,y) and \Theta(x,y) from a desired function Z_1(x,y)\Theta_1(x,y). This is done for blazed and
sinusiodal groove profiles.
"""

bla_x_curve = numpy.ogrid[1e-9:bla_max_inv:1000j]
bla_response = ch.curve_arbitrary(bla_x_curve,ch.cbla,eta,matches_sorted)
bla_Z_curve, bla_A_curve = numpy.abs(bla_response), numpy.unwrap(numpy.angle(bla_response))

sin_x_curve = numpy.ogrid[1e-9:sin_max_inv:1000j]
sin_response = ch.curve_sinusoidal(sin_x_curve,eta)
sin_Z_curve, sin_A_curve = numpy.abs(sin_response), numpy.unwrap(numpy.angle(sin_response))

# %% Code Cell ##################################################################

"""
Inverted curves of Z_1 and \Theta_1 as a functions of Z for blazed and sinusoidal groove profiles
"""

fig,(ax1,ax2) = pyplot.subplots(1,2,figsize=(6,3),dpi=300)
ax1.plot(bla_Z_curve, bla_x_curve)
ax1.set_xlabel('$Z_1$')
ax1.set_ylabel('$hZ$')
ax1.set_title('Blazed grooves')
ax2.plot(bla_A_curve, bla_x_curve)
ax2.set_xlabel(r'$\Theta_1$')
ax2.set_ylabel('$hZ$')
ax2.set_title('Blazed grooves')
pyplot.subplots_adjust(wspace=0.35)
pyplot.show()

fig,(ax1,ax2) = pyplot.subplots(1,2,figsize=(6,3),dpi=300)
ax1.plot(sin_Z_curve, sin_x_curve)
ax1.set_xlabel('$Z_1$')
ax1.set_ylabel('$hZ$')
ax1.set_title('Sinusoidal grooves')
ax2.plot(sin_A_curve, sin_x_curve)
ax2.set_xlabel(r'$\Theta_1$')
ax2.set_ylabel('$hZ$')
ax2.set_title('Sinusoidal grooves')
pyplot.subplots_adjust(wspace=0.35)
pyplot.show()

# %% Code Cell ##################################################################

"""
Function desired to reconstruct with hologram, here is an LG_3^5 example and the corrected hologram for blazed,
sinusoidal, and binary groove profiles
"""

num = 512                                             # array pixel dimension
size = 30.                                            # array width in um
grid = numpy.ogrid[size/2:-size/2:num*1j, -size/2:size/2:num*1j]
y, x = numpy.broadcast_arrays(*grid)                  # y,x linear meshgrids
m1 = 5                                                # azimuthal index
p1 = 3                                                # radial index
w0 = 3.25                                             # gaussian beam waist in um
lg_beam = ch.LG(x, y, m1, p1, w0)                     # complex LG_3^5 wavefunction
lg_beam /= numpy.max(numpy.abs(lg_beam))              # max amp. normalized complex wavefunction
pitch = w0*0.4/1.5 # grating pitch

# calculate grating
bla_grating = ch.correct_grating_arbitrary(lg_beam, bla_max_inv, x, bla_Z_curve, bla_A_curve, bla_x_curve, ch.cbla, 199, pitch)
sin_grating = ch.correct_grating_sinusoidal(lg_beam, sin_max_inv, x, sin_Z_curve, sin_A_curve, sin_x_curve, 199, pitch)
bin_grating = ch.correct_grating_binary(lg_beam,size,199,pitch,sc=1.00)

# pad grating
px = 2**13
px2 = 2**12
bla_grat = numpy.zeros((px,px))
bla_grat[px2-num//2:px2+num//2,px2-num//2:px2+num//2] = bla_grating
sin_grat = numpy.zeros((px,px))
sin_grat[px2-num//2:px2+num//2,px2-num//2:px2+num//2] = sin_grating
bin_grat = numpy.zeros((px,px))
bin_grat[px2-num//2:px2+num//2,px2-num//2:px2+num//2] = bin_grating

# calculate back focal plane of each grating
bla_psi = numpy.exp(1.0j*bla_grat*eta*bla_max_inv)
bla_Psi = ch.fft2(bla_psi)
sin_psi = numpy.exp(1.0j*sin_grat*eta*sin_max_inv)
sin_Psi = ch.fft2(sin_psi)
bin_psi = numpy.exp(1.0j*sin_grat*eta*bin_max_inv)
bin_Psi = ch.fft2(bin_psi)

# %% Code Cell ##################################################################

"""
Complex intensity image of simulated LG_3^5 mode, corrected holograms, and the first diffraction order in the back focal
plane of the hologram for blazed, sinusoidal, and binary groove profiles.
"""

xsh = -555              # pixel location of first diffraction order in back focal plane
xwid = 200              # 1/2 width of window in image of back focal plane
ywid = 200              # 1/2 height of window in image of back focal plane

fig,(ax1,ax2,ax3) = pyplot.subplots(1,3,dpi=300,figsize=(9,3))
im1 = ax1.imshow(ch.image(lg_beam,norm=1),cmap=ch.phs_cmap())
ax1.set_title(r'$|$LG$_3^5|^2$')
ax1.set_xticks([])
ax1.set_yticks([])
cbar1 = fig.colorbar(im1,ax=ax1,pad=0.02,shrink=0.75)
cbar1.set_ticks(numpy.arange(0, 2**8, 2**7-0.5).astype('int'))
cbar1.set_ticklabels([r'$-\pi$',r'$0$',r'$\pi$'])
im2 = ax2.imshow(ch.image_real(bla_grating),cmap=ch.copper_cmap())
ax2.set_title(r'Blazed corrected hologram')
ax2.set_xticks([])
ax2.set_yticks([])
cbar2 = fig.colorbar(im2,ax=ax2,pad=0.02,shrink=0.75)
cbar2.set_ticks([0,255])
cbar2.set_ticklabels([0,'h'])
im3 = ax3.imshow(ch.image(bla_Psi[px2-ywid:px2+ywid,px2+xsh-xwid:px2+xsh+xwid],norm=1),cmap=ch.phs_cmap())
ax3.set_title(r'Back focal plane')
ax3.set_xticks([])
ax3.set_yticks([])
cbar3 = fig.colorbar(im3,ax=ax3,pad=0.02,shrink=0.75)
cbar3.set_ticks(numpy.arange(0, 2**8, 2**7-0.5).astype('int'))
cbar3.set_ticklabels([r'$-\pi$',r'$0$',r'$\pi$'])
pyplot.subplots_adjust(wspace=0.1)
pyplot.show()

fig,(ax1,ax2,ax3) = pyplot.subplots(1,3,dpi=300,figsize=(9,3))
im1 = ax1.imshow(ch.image(lg_beam,norm=1),cmap=ch.phs_cmap())
ax1.set_title(r'$|$LG$_3^5|^2$')
ax1.set_xticks([])
ax1.set_yticks([])
cbar1 = fig.colorbar(im1,ax=ax1,pad=0.02,shrink=0.75)
cbar1.set_ticks(numpy.arange(0, 2**8, 2**7-0.5).astype('int'))
cbar1.set_ticklabels([r'$-\pi$',r'$0$',r'$\pi$'])
im2 = ax2.imshow(ch.image_real(sin_grating),cmap=ch.copper_cmap())
ax2.set_title(r'Sinusoidal corrected hologram')
ax2.set_xticks([])
ax2.set_yticks([])
cbar2 = fig.colorbar(im2,ax=ax2,pad=0.02,shrink=0.75)
cbar2.set_ticks([0,255])
cbar2.set_ticklabels([0,'h'])
im3 = ax3.imshow(ch.image(sin_Psi[px2-ywid:px2+ywid,px2+xsh-xwid:px2+xsh+xwid],norm=1),cmap=ch.phs_cmap())
ax3.set_title(r'Back focal plane')
ax3.set_xticks([])
ax3.set_yticks([])
cbar3 = fig.colorbar(im3,ax=ax3,pad=0.02,shrink=0.75)
cbar3.set_ticks(numpy.arange(0, 2**8, 2**7-0.5).astype('int'))
cbar3.set_ticklabels([r'$-\pi$',r'$0$',r'$\pi$'])
pyplot.subplots_adjust(wspace=0.1)
pyplot.show()

fig,(ax1,ax2,ax3) = pyplot.subplots(1,3,dpi=300,figsize=(9,3))
im1 = ax1.imshow(ch.image(lg_beam,norm=1),cmap=ch.phs_cmap())
ax1.set_title(r'$|$LG$_3^5|^2$')
ax1.set_xticks([])
ax1.set_yticks([])
cbar1 = fig.colorbar(im1,ax=ax1,pad=0.02,shrink=0.75)
cbar1.set_ticks(numpy.arange(0, 2**8, 2**7-0.5).astype('int'))
cbar1.set_ticklabels([r'$-\pi$',r'$0$',r'$\pi$'])
im2 = ax2.imshow(ch.image_real(bin_grating),cmap=ch.copper_cmap())
ax2.set_title(r'Binary corrected hologram')
ax2.set_xticks([])
ax2.set_yticks([])
cbar2 = fig.colorbar(im2,ax=ax2,pad=0.02,shrink=0.75)
cbar2.set_ticks([0,255])
cbar2.set_ticklabels([0,'h'])
im3 = ax3.imshow(ch.image(bin_Psi[px2-ywid:px2+ywid,px2+xsh-xwid:px2+xsh+xwid],norm=1),cmap=ch.phs_cmap())
ax3.set_title(r'Back focal plane')
ax3.set_xticks([])
ax3.set_yticks([])
cbar3 = fig.colorbar(im3,ax=ax3,pad=0.02,shrink=0.75)
cbar3.set_ticks(numpy.arange(0, 2**8, 2**7-0.5).astype('int'))
cbar3.set_ticklabels([r'$-\pi$',r'$0$',r'$\pi$'])
pyplot.subplots_adjust(wspace=0.1)
pyplot.show()
