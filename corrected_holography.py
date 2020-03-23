
"""
corrected_holography.py python functions for corrected off-axis hologram pattern generation.

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
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy
import copy
import heapq
import gc
import itertools
from matplotlib.colors import LinearSegmentedColormap
import scipy.special
import os
import time

"""
This block of functions are used to extract the maps s_j(n) that are of a specified order ranked and sorted by their contribution j.
"""

def cbla(n):
    """Fourier coefficients for a blazed profile"""
    if n == 0:
        return 0.5
    return 1.j / (2 * numpy.pi * n)


def cbin(n,b):
    """Fourier coefficients for a binary profile"""
    if n==0:
        return b
    else:
        return numpy.sin(n*numpy.pi*b)/(n*numpy.pi)


def component(Z, n, sn, c, eta):
    """
    The operand of the sum in equation 9

    Z: value of Z(r) to use.  Can be a numpy 1d array
    n: term of the infinite product
    sn: s(n) for a particular s
    c: function generator for the Fourier coefficients
    eta: material parameters
    """
    if n == 0:
        return numpy.exp(-1.j * eta * Z * c(0))
    return numpy.exp(1.j * numpy.angle(c(n)) * sn) * scipy.special.iv(sn, -2.j * eta * Z * numpy.abs(c(n)))


class memory(list):
    """
    Convenience class so that

    @memory
    def func(*args, **kwargs)
        ...
        ...

    res = func(args)
    res == func[0] ----> True
    """
    def __init__(self, func):
        super(memory, self).__init__()
        self.func = func
        self.append(None)

    def __call__(self, *args, **kwargs):
        self[0] = self.func(*args, **kwargs)
        return self[0]


@memory
def val(sequence, Z, eta, c):
    """Returns the absolue value of the product of components to rank a given sequence"""
    return numpy.abs(numpy.prod([numpy.sum(component(Z, pair[0], pair[1], c, eta)) for pair in sequence])) / 10


@memory
def order(sequence):
    """Returns the order of a sequence as sum(n * m)"""
    return sum([pair[0] * pair[1] for pair in sequence])

def sign(x):
    """Returns sign(x) = { -1 if x<0; 1 if x>=0 }"""
    if x == 0:
        return 1
    else:
        return numpy.sign(x)


def extend(a, b, level):
    """Returns array of tuples appended with the tuple b, a = [[a1,a2],[a3,a4],...,[b1,b2]]"""
    if level > 1:
        try:
            a.extend(b[0])
        except:
            a.append(b)
    else:
        a.append(b)
    return a


def Search(N, val, level, start_level, minimum, r):
    """Generator for the function search(*args)"""
    n = N
    forward = False
    sig = N
    s = 1
    if level == 1:
        if not abs(r) == 0:
            s = abs(r)
        while n > 0:
            for i in range(-s, s + 1):
                if i == 0:
                    continue
                if val + i * n == 1:
                    forward = True
                    yield [[n, i], ['end']]
            n -= 1
    else:
        while n > 0:
            if level == start_level:
                if n < minimum:
                    break
            sig = 0.5 * (1 + (level)) * (2 * (n - 1) - (level)) + 2
            if sig < abs(val - 2 * n) <= 2 * sig:
                forward = True
                yield extend([[n, -2]],
                             Search(n - 1, val - 2 * n, level - 1, start_level, minimum, r), level - 1)
            if abs(val - n) <= sig:
                forward = True
                yield extend([[n, -1]],
                             Search(n - 1, val - n, level - 1, start_level, minimum, r), level - 1)
            if abs(val + n) <= sig:
                forward = True
                yield extend([[n,  1]],
                             Search(n - 1, val + n, level - 1, start_level, minimum, r), level - 1)
            if sig < abs(val + 2 * n) <= 2 * sig:
                forward = True
                yield extend([[n,  2]],
                             Search(n - 1, val + 2 * n, level - 1, start_level, minimum, r), level - 1)
            n -= 1
    if forward is False:
        yield [[n,  sign(val), level, val + sign(val) * n, n, sig, 'ended', r]]


def retreive(t, cur, tree):
    """Retrieves search results from generator Search(*args)"""
    try:
        for item in list(tree):
            val = copy.copy(cur)
            if len(item) > 1 and str(item[1].__class__) == "<class 'generator'>":
                val.append(item[0])
                retreive(t, val, list(item[1]))
            else:
                val.append(item[0])
                t.append(val)
    except:
        pass


def search(N, level, ext, begin, r=1):
    """
    This is used to find more unordered and unranked maps s(n) not found from the small parameter brute force search

    N:      maximum value of n to search to
    level:  number of non-zero values in the maps s(n)
    ext:    set all values up to s(ext) to zero for all n > the last non-zero entry of s(n)
    begin:  the minimum value of n that can contains the last non-zero entry of s(n)
    r:      the maximum value |s(n)| can be
    """
    if ext < N:
        raise AttributeError("'ext' must be greater than 'N'")
    b = list(Search(N, 0, level, level, begin, r))
    t = []
    retreive(t, [], b)

    tmp = [item for item in t if len(item) == level and order(item) == 1 and len(item[-1]) == 2]
    ret = []
    for item in tmp:
        ar = numpy.array(item)
        temp = [[i, item[list(ar[::,0]).index(i)][1] if i in ar[::,0] else 0] for i in range(1, ext + 1)]
        ret.append(temp)
    return ret


def first_order_contributions(coefficients, eta, height, p, q, ord, fpath, fname, limit_search = False):
    """
    Generate a large sequence of terms which contribute to the first diffraction order,
    and also calculate the contribution from each and saves to file

    coefficients: Fourier coefficients for groove profile
    eta:          material parameter for longitudinal phase shift and attenuation per unit length
    height:       maximum pattern depth
    p:            maximum s(n) value for brute force search
    q:            maximum n value for brute force search
    fpath:        file path for save file
    fname:        file name for save file
    """
    N = 100                                         # maximum value of n to search to
    Z = numpy.ogrid[0:height:5.j]                   # a sparse array to evaluate for each term
    rr = [[i, 0] for i in range(q + 1, N + 1)]      # This is to extend the sequences we get to the max n
    r = (list(l) + rr for l in itertools.product(*[[[n, m] for m in range(-p, p + 1)] for n in range(1,  q + 1)]))

    t0 = time.time()
    pairs = search(N, 2, N, q + 1, 19)

    if not limit_search:
        print("1/6 search calls completed. {:.4f}s elapsed. ".format(time.time()-t0))
        triples = search(int(N / 2), 3, N, q + 1, 7)
        print("2/6 search calls completed. {:.4f}s elapsed. ".format(time.time()-t0))
        quads = search(int(N / 3), 4, N, q + 1, 5)
        print("3/6 search calls completed. {:.4f}s elapsed. ".format(time.time()-t0))
        pentas = search(int(N / 4), 5, N, q + 1, 5)
        print("4/6 search calls completed. {:.4f}s elapsed. ".format(time.time()-t0))
        hexas = search(int(N / 5), 6, N, q + 1, 5)
        print("5/6 search calls completed. {:.4f}s elapsed. ".format(time.time()-t0))
        septas = search(int(N / 6), 7, N, q + 1, 5)
        print("6/6 search calls completed. {:.4f}s elapsed. ".format(time.time()-t0))
        print("Patterns Initiated, adding {} terms.".format(len(pairs) + len(triples) + len(quads) + len(pentas)
                                                           + len(hexas) + len(septas)))
        matches = {i: {'list': list(sequence),
                       'order': order[0],
                       'val': val[0]}
                       for i, sequence in enumerate(itertools.chain(r, pairs, triples, quads, pentas, hexas, septas))
                       if order(sequence) == ord and val(list(sequence), Z, eta, coefficients) > 0}

    elif limit_search:
        print("1/1 search calls completed. {:.4f}s elapsed. ".format(time.time()-t0))
        print("Patterns Initiated, adding {} terms.".format(len(pairs)))

        matches = {i: {'list': list(sequence),
                       'order': order[0],
                       'val': val[0]}
                       for i, sequence in enumerate(itertools.chain(r, pairs))
                       if order(sequence) == ord and val(list(sequence), Z, eta, coefficients) > 0}

    print("Found {} contributors to the first diffraction order".format(len(matches)))

    gc.collect()

    sorted = {i: val[1] for i, val in enumerate(heapq.nlargest(50000, matches.items(), key=lambda item: item[1]['val']))}

    if os.path.exists(fpath):
        pass
    else:
        os.mkdir(fpath)

    numpy.save('{}/{}.npy'.format(fpath,fname),sorted)

    return sorted


"""
This block of functions are used to generate a corrected hologram groove profile from the set of calculated maps s_j(n).
"""

def curve_arbitrary(x,c,eta,matches):
    f = component(x, 0, 0, c, eta) * numpy.sum([numpy.prod([component(x, k[0], k[1], c, eta)
                                                for k in matches[i]['list']], axis=0)
                                                for i in range(min(5000, len(matches)))], axis=0)
    return f/numpy.max(numpy.abs(f))


def curve_sinusoidal(x,eta):
    f = numpy.exp(-0.5j*eta*x)*scipy.special.iv(1,-0.5j*eta*x)
    return f/numpy.max(numpy.abs(f))


def LP(x,m,p):
    lpm = [numpy.ones_like(x),1+m-x]
    if p>1:
        for i in range(p-1):
            lpm.append(((2*(i+1)+1+m-x)*lpm[-1]-(i+1+m)*lpm[-2])/(i+2))
        return lpm[p]
    else:
        return lpm[p]


def LG(X, Y, m, p, waist, phs=0):
    R = numpy.sqrt((X ** 2 + Y ** 2))*numpy.sqrt(2)/waist
    phi = numpy.arctan2(Y, X)

    return (numpy.sqrt(2*numpy.math.factorial(p)/(numpy.pi*numpy.math.factorial(p+abs(m)))) *
            R**numpy.abs(m)*numpy.exp(-R**2/2)*LP(R**2,numpy.abs(m),p)/waist *
            numpy.exp(-1.j * (m * phi + phs)))


def correct_grating_arbitrary(beam, height, X, Z_curve, A_curve, x_curve, coefficients, num, pitch):
    peak = x_curve[-1]
    c = x_curve[-1] / height
    Z_curve /= Z_curve[-1]
    Z_array = numpy.interp(numpy.abs(beam), c * Z_curve, x_curve) / numpy.interp(1, c * Z_curve, x_curve)
    arg = numpy.unwrap(numpy.angle(beam) + numpy.interp(height * Z_array, x_curve, A_curve))

    run = numpy.zeros_like(arg, dtype=numpy.float)
    run += coefficients(0)
    for n in range(1, num + 1):
        cn = coefficients(n)
        if not numpy.isclose(cn, 0):
            run += 2 * (numpy.abs(cn) *
                        numpy.cos(n * (2 * numpy.pi / pitch * X - arg) + numpy.angle(cn)))
    return run * Z_array

def correct_grating_sinusoidal(beam, height, X, Z_curve, A_curve, x_curve, pitch):
    peak = x_curve[-1]
    c = x_curve[-1] / height
    Z_curve /= Z_curve[-1]
    Z_array = numpy.interp(numpy.abs(beam), c * Z_curve, x_curve) / numpy.interp(1, c * Z_curve, x_curve)
    arg = numpy.unwrap(numpy.angle(beam) + numpy.interp(height * Z_array, x_curve, A_curve))

    run = numpy.zeros_like(arg, dtype=numpy.float)
    run += 0.5 * ( 1 + numpy.cos( 2 * numpy.pi / pitch * X - arg ) )
    return run * Z_array

def correct_grating_binary(beam,size,num,pitch,sc=1.00):
    br = numpy.arcsin(numpy.abs(beam)/numpy.max(numpy.abs(beam)))/numpy.pi
    arg = numpy.angle(beam) - numpy.pi/2
    x = numpy.linspace(-size//2,size//2,numpy.shape(beam)[0])
    X,Y = numpy.meshgrid(x,x)

    run = numpy.zeros_like(arg, dtype=numpy.float)
    run += cbin(0,sc*br)
    for n in range(1, num + 1):
        cn = cbin(n,sc*br)
        run += 2 * cn * numpy.cos(n * (2 * numpy.pi / pitch * X - arg) )
    return run


"""
This block contains functions that are used to transform complex numpy arrays to a pyplot plottable RBG array where brightness represents amplitude and color represents phase, as well as a print output table function for an ordered and ranked set of s(n)'s.
"""

copper = numpy.array([0, 0, 0, 1, 0, 0, 2, 1, 0, 3, 2, 1, 4, 3, 1, 6, 3, 2, 7, 4, 2, 8, 5, 3, 9, 6, 3, 11, 7, 4, 12, 7, 4, 13, 8, 5, 14, 9, 5, 16, 10, 6, 17, 10, 6, 18, 11, 7, 19, 12, 7, 20, 13, 8, 22, 14, 8, 23, 14, 9, 24, 15, 9, 25, 16, 10, 27, 17, 10, 28, 17, 11, 29, 18, 11, 30, 19, 12, 32, 20, 12, 33, 21, 13, 34, 21, 13, 35, 22, 14, 37, 23, 14, 38, 24, 15, 39, 24, 15, 40, 25, 16, 41, 26, 16, 43, 27, 17, 44, 28, 17, 45, 28, 18, 46, 29, 18, 48, 30, 19, 49, 31, 19, 50, 32, 20, 51, 32, 20, 53, 33, 21, 54, 34, 21, 55, 35, 22, 56, 35, 22, 58, 36, 23, 59, 37, 23, 60, 38, 24, 61, 39, 24, 62, 39, 25, 64, 40, 25, 65, 41, 26, 66, 42, 26, 67, 42, 27, 69, 43, 27, 70, 44, 28, 71, 45, 28, 72, 46, 29, 74, 46, 29, 75, 47, 30, 76, 48, 30, 77, 49, 31, 79, 49, 31, 80, 50, 32, 81, 51, 32, 82, 52, 33, 83, 53, 33, 85, 53, 34, 86, 54, 34, 87, 55, 35, 88, 56, 35, 90, 57, 36, 91, 57, 36, 92, 58, 37, 93, 59, 37, 95, 60, 38, 96, 60, 38, 97, 61, 39, 98, 62, 39, 100, 63, 40, 101, 64, 40, 102, 64, 41, 103, 65, 41, 104, 66, 42, 106, 67, 42, 107, 67, 43, 108, 68, 43, 109, 69, 44, 111, 70, 44, 112, 71, 45, 113, 71, 45, 114, 72, 46, 116, 73, 46, 117, 74, 47, 118, 74, 47, 119, 75, 48, 121, 76, 48, 122, 77, 49, 123, 78, 49, 124, 78, 50, 125, 79, 50, 127, 80, 51, 128, 81, 51, 129, 82, 52, 130, 82, 52, 132, 83, 53, 133, 84, 53, 134, 85, 54, 135, 85, 54, 137, 86, 55, 138, 87, 55, 139, 88, 56, 140, 89, 56, 142, 89, 57, 143, 90, 57, 144, 91, 58, 145, 92, 58, 146, 92, 59, 148, 93, 59, 149, 94, 60, 150, 95, 60, 151, 96, 61, 153, 96, 61, 154, 97, 62, 155, 98, 62, 156, 99, 63, 158, 99, 63, 159, 100, 64, 160, 101, 64, 161, 102, 65, 163, 103, 65, 164, 103, 66, 165, 104, 66, 166, 105, 67, 167, 106, 67, 169, 107, 68, 170, 107, 68, 171, 108, 69, 172, 109, 69, 174, 110, 70, 175, 110, 70, 176, 111, 71, 177, 112, 71, 179, 113, 72, 180, 114, 72, 181, 114, 73, 182, 115, 73, 184, 116, 74, 185, 117, 74, 186, 117, 75, 187, 118, 75, 188, 119, 76, 190, 120, 76, 191, 121, 77, 192, 121, 77, 193, 122, 78, 195, 123, 78, 196, 124, 79, 197, 124, 79, 198, 125, 80, 200, 126, 80, 201, 127, 81, 202, 128, 81, 203, 128, 82, 205, 129, 82, 206, 130, 83, 207, 131, 83, 208, 132, 84, 209, 132, 84, 211, 133, 85, 212, 134, 85, 213, 135, 86, 214, 135, 86, 216, 136, 87, 217, 137, 87, 218, 138, 88, 219, 139, 88, 221, 139, 89, 222, 140, 89, 223, 141, 90, 224, 142, 90, 226, 142, 91, 227, 143, 91, 228, 144, 92, 229, 145, 92, 230, 146, 93, 232, 146, 93, 233, 147, 94, 234, 148, 94, 235, 149, 95, 237, 149, 95, 238, 150, 96, 239, 151, 96, 240, 152, 97, 242, 153, 97, 243, 153, 98, 244, 154, 98, 245, 155, 99, 247, 156, 99, 248, 157, 99, 249, 157, 100, 250, 158, 100, 251, 159, 101, 253, 160, 101, 254, 160, 102, 255, 161, 102, 255, 162, 103, 255, 163, 103, 255, 164, 104, 255, 164, 104, 255, 165, 105, 255, 166, 105, 255, 167, 106, 255, 167, 106, 255, 168, 107, 255, 169, 107, 255, 170, 108, 255, 171, 108, 255, 171, 109, 255, 172, 109, 255, 173, 110, 255, 174, 110, 255, 174, 111, 255, 175, 111, 255, 176, 112, 255, 177, 112, 255, 178, 113, 255, 178, 113, 255, 179, 114, 255, 180, 114, 255, 181, 115, 255, 182, 115, 255, 182, 116, 255, 183, 116, 255, 184, 117, 255, 185, 117, 255, 185, 118, 255, 186, 118, 255, 187, 119, 255, 188, 119, 255, 189, 120, 255, 189, 120, 255, 190, 121, 255, 191, 121, 255, 192, 122, 255, 192, 122, 255, 193, 123, 255, 194, 123, 255, 195, 124, 255, 196, 124, 255, 196, 125, 255, 197, 125, 255, 198, 126, 255, 199, 126])

def image_real(data):
    """
    This will clip the data such that 0 <= data <= 1, and display only this portion.
    Normalize the data before calling this function.
    """

    split_view = numpy.dtype(
        (numpy.uint32,
         {'r': (numpy.uint8, 0), 'g': (numpy.uint8, 1),
          'b': (numpy.uint8, 2), 'a': (numpy.uint8, 3)})
    )
    data = data[::]
    numpy.putmask(data, data < 0, 0)
    numpy.putmask(data, data > 1, 1)
    data = numpy.floor(255 * data).astype(numpy.uint32)
    alpha = numpy.full_like(data, fill_value=255, dtype=numpy.uint8)
    array = numpy.zeros_like(data, dtype=numpy.uint32)
    array = numpy.require(array, requirements=['A', 'O', 'W', 'C'])

    array_rgb = array.view(split_view)
    array_rgb['r'] = copper[0::3][data].astype(numpy.uint8)
    array_rgb['g'] = copper[1::3][data].astype(numpy.uint8)
    array_rgb['b'] = copper[2::3][data].astype(numpy.uint8)
    array_rgb['a'] = alpha
    return numpy.frombuffer(array.tobytes(), dtype=numpy.uint8).reshape(array.shape[0], array.shape[1], 4)


def normalize_array(array):
    """Returns a copy of the array such that 0 <= array <= 1"""
    maximum = numpy.max(numpy.abs(array))
    if numpy.isclose(maximum, 0):
        return array / maximum
    return array / maximum


def image(data, norm=0):
    """
    This will return a QImage from the array 'data':
        norm = 0 ---->  data -> data
        norm = 1 ---->  data -> data ** 2
        norm = 2 ---->  data -> log(1 + data) / log(2)
        norm = 3 ---->  data -> real image w/ copper tone
    """
    if norm==3:
        narray = normalize_array(numpy.abs(data))
        return image_real(narray)
    else:
        split_view = numpy.dtype(
            (numpy.uint32,
             {'r': (numpy.uint8, 0), 'g': (numpy.uint8, 1),
              'b': (numpy.uint8, 2), 'a': (numpy.uint8, 3)})
        )
        alpha = numpy.full_like(data, fill_value=255, dtype=numpy.uint8)
        data = normalize_array(data)
        val = numpy.absolute(data)
        array = numpy.zeros_like(data, dtype=numpy.uint32)
        array = numpy.require(array, requirements=['A', 'O', 'W', 'C'])
        array_rgb = array.view(split_view)

        if norm == 0:
            pass
        elif norm == 1:
            val = numpy.square(val)
        elif norm == 2:
            val = numpy.log(val + 1) / (numpy.log(2))

        val *= 255
        hue = (numpy.unwrap(numpy.angle(data)) + numpy.pi) / 2
        pi6 = numpy.pi / 6


        g = 0.6 * numpy.sin(hue - 2 * pi6)**2
        r = numpy.sin(hue - 0.15 * pi6)**2
        r += 0.35 * numpy.sin(hue - 3.15 * pi6)**2
        g += 0.065 * numpy.sin(hue - 5.05 * pi6)**2
        b = numpy.sin(hue - 4.25 * pi6)**2

        g += 0.445 * b
        g += 0.33 * r


        array_rgb['r'] = (r * val).astype(numpy.uint8)
        array_rgb['g'] = (g * val).astype(numpy.uint8)
        array_rgb['b'] = (b * val).astype(numpy.uint8)
        array_rgb['a'] = alpha
        return numpy.frombuffer(array.tobytes(), dtype=numpy.uint8).reshape(array.shape[0], array.shape[1], 4)


def phs_cmap():
    hue = numpy.linspace(0,numpy.pi)
    pi6 = numpy.pi / 6

    g = 0.6 * numpy.sin(hue - 2 * pi6)**2
    r = numpy.sin(hue - 0.15 * pi6)**2
    r += 0.35 * numpy.sin(hue - 3.15 * pi6)**2
    g += 0.065 * numpy.sin(hue - 5.05 * pi6)**2
    b = numpy.sin(hue - 4.25 * pi6)**2

    g += 0.445 * b
    g += 0.33 * r

    colors = numpy.transpose([r,g,b])
    n_bin = 300  # Discretizes the interpolation into bins
    cmap_name = 'my_list'

    return LinearSegmentedColormap.from_list(
            cmap_name, colors, N=n_bin)


def copper_cmap():
    colors = numpy.transpose([copper[0::3]/255.,copper[1::3]/255.,copper[2::3]/255.])
    n_bin = 300  # Discretizes the interpolation into bins
    cmap_name = 'my_list'

    return LinearSegmentedColormap.from_list(
            cmap_name, colors, N=n_bin)


def print_sns_table(sns,num=25,ji=1,jf=100):
    """
    Prints text output table of s_j(n) up to n=num, ordered by j over the range of j in [ji,jf].
    """
    form = "{0}|" + "".join(("{"+str(n)+ "}" for n in range(1,num+1))) + " ... || {"+str(num+1)+":8.8f}"
    print(form[:-10].format("n  ".rjust(10), *[str(i + 1).rjust(3) for i in range(num)], ""), "Relative Value")
    print(" =========|" + "=" * (3 * num + 5) + "||" + "-" * 15)
    for i in range(ji, jf+1):
        print(form.format(("s_"+str(i)+"(n)").rjust(10), *[str(m).rjust(3) for n, m in sns[i-1]['list'][0:num]],sns[i-1]['val'] / sns[0]['val']))
    for i in range(3):
        print(".  |".rjust(11)+" "*int(num*1.5)+"."+" "*round(num*1.5+5-(num-1)%2)+'||    .')


def fft2(im):
    return numpy.fft.ifftshift(numpy.fft.fft2(numpy.fft.fftshift(im)))


def ifft2(im):
    return numpy.fft.fftshift(numpy.fft.ifft2(numpy.fft.ifftshift(im)))
