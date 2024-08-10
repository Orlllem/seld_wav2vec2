import random

import numpy as np
cimport cython

from math import pi
from scipy.ndimage.interpolation import shift
from seld_ambisonics.common import CHANNEL_ORDERING, NORMALIZATION, DEFAULT_ORDERING, DEFAULT_NORMALIZATION, DEFAULT_ORDER, DEFAULT_RADIUS, DEFAULT_RATE
from scipy.signal import resample
from scipy.special import lpmv
cimport numpy as cnp
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d


# Phi can vary all around the horizontal plane, so 0 360
# Elevation is common to vary between -60 60, but we can choose different values as well,such as -90 90
PHI, ELE, Z = np.arange(0, 360, 5), np.arange(-60, 60, 5), np.arange(0, 10, 0.2)

cnp.import_array()

cdef extern from "math.h":
    cpdef double sin(double x)
    cpdef double cos(double x)
    cpdef double atan2(double x, double y)
    cpdef double sqrt(double x)

    
cpdef int factorial(int n):
    cdef int i, ret
    ret = 1
    for i in range(n):
        ret *= n
    return ret    

    
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def max_norm(int n, int m):
        assert n <= 3
        if n == 0:
            return 1/sqrt(2.)
        elif n == 1:
            return 1.
        elif n == 2:
            return 1. if m == 0 else 2. / sqrt(3.)
        else:
            return 1. if m == 0 else (sqrt(45. / 32) if m in [1, -1] else 3. / sqrt(5.))


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def sn3d_norm(int n, int m):
    return sqrt((2. - float(m == 0)) * float(factorial(n-abs(m))) / float(factorial(n+abs(m))))


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def n3d_norm(int n, int m):
    return sn3d_norm(n, m) * sqrt((2*n+1) / (4.*pi))


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def normalization_factor(int index):
    # assert ordering in CHANNEL_ORDERING
    # assert normalization in NORMALIZATION

    order, degree = index_to_degree_order(index)
    return sn3d_norm(order, degree)

    
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def convert_ordering(int index, int orig_ordering, int dest_ordering):
    # assert orig_ordering in CHANNEL_ORDERING
    # assert dest_ordering in CHANNEL_ORDERING
    if dest_ordering == orig_ordering:
        return index

    n, m = index_to_degree_order(index, orig_ordering)
    return degree_order_to_index(n, m, dest_ordering)
    

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def index_to_degree_order(int index):
    # assert ordering in CHANNEL_ORDERING
    cdef int order = int(sqrt(index))
    index -= order**2

    cdef int degree = index - order
    return order, degree
 

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def acn_idx(int n, int m):
    return n*(n+1)+m  


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def degree_order_to_index(int order, int degree):
    # assert -order <= degree <= order
    # assert ordering in CHANNEL_ORDERING

    return acn_idx(order, degree)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def spherical_harmonic_mn(int order, int degree, double phi, double nu):
    norm = normalization_factor(degree_order_to_index(order, degree))
    sph = (-1)**degree * norm * \
        lpmv(abs(degree), order, sin(nu)) * \
        (cos(abs(degree) * phi) if degree >= 0 else sin(abs(degree) * phi))
    return sph


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def spherical_harmonics(position, int max_order):
    # assert isinstance(position, Position)

    cdef int num_channels = int((max_order+1)**2)
    cdef cnp.ndarray output = np.zeros((num_channels,))
    for i in range(num_channels):
        order, degree = index_to_degree_order(i)
        output[i] = spherical_harmonic_mn(
            order, degree, position.phi, position.nu)
    return output


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def spherical_harmonics_matrix(positions, int max_order):
#     assert isinstance(positions, list) and all(
#         [isinstance(p, Position) for p in positions])

    cdef int num_channels = int((max_order + 1) ** 2)
    cdef cnp.ndarray[float, ndim=2] Y = np.zeros((len(positions), num_channels), dtype = np.float32)
    for i, p in enumerate(positions):
        Y[i] = spherical_harmonics(p, max_order)
    return Y


class AmbiFormat(object):
    def __init__(self,
                 ambi_order=DEFAULT_ORDER,
                 sample_rate=DEFAULT_RATE,
                 radius=DEFAULT_RADIUS,
                 ordering=DEFAULT_ORDERING,
                 normalization=DEFAULT_NORMALIZATION):
        self.order = ambi_order
        self.num_channels = int((ambi_order+1)**2)
        self.radius = radius
        self.sample_rate = sample_rate
        self.ordering = ordering
        self.normalization = normalization


class Position(object):
    def __init__(self, double x1, double x2, double x3, c_type):
        assert c_type.lower() in ['cartesian', 'polar']

        self.x, self.y, self.z = 0., 0., 0.
        self.phi, self.nu, self.r = 0., 0., 0.
        if c_type == 'cartesian':
            self.set_cartesian(x1, x2, x3)
        else:
            self.set_polar(x1, x2, x3)

    def clone(self):
        return Position(self.x, self.y, self.z, 'cartesian')

    def set_cartesian(self, double x, double y, double z):
        self.x, self.y, self.z = x, y, z
        self.calc_polar()
        self.calc_cartesian()

    def set_polar(self, double phi, double nu, double r):
        self.phi, self.nu, self.r = phi, nu, r
        self.calc_cartesian()
        self.calc_polar()

    def calc_cartesian(self):
        self.x = self.r * cos(self.phi) * cos(self.nu)
        self.y = self.r * sin(self.phi) * cos(self.nu)
        self.z = self.r * sin(self.nu)

    def calc_polar(self):
        self.phi = atan2(self.y, self.x)
        self.nu = atan2(self.z, sqrt(self.x**2+self.y**2))
        self.r = sqrt(self.x**2+self.y**2+self.z**2)

    def rotate(self, cnp.ndarray[float] rot_matrix):
        pos = np.dot(rot_matrix, np.array([self.x, self.y, self.z]).reshape(3, 1))
        self.x, self.y, self.z = pos[0], pos[1], pos[2]
        self.calc_polar()
        self.calc_cartesian()

    def set_radius(self, double radius):
        self.r = radius
        self.calc_cartesian()

    def coords(self, c_type):
        if c_type == 'cartesian':
            return np.array([self.x, self.y, self.z])
        elif c_type == 'polar':
            return np.array([self.phi, self.nu, self.r])
        else:
            raise ValueError('Unknown coordinate type. Use cartesian or polar.')

    def print_position(self, c_type=None):
        if c_type is None or c_type == 'cartesian':
            print('Cartesian (x,y,z): (%.2f, %.2f, %.2f)' % (self.x, self.y, self.z))
        if c_type is None or c_type == 'polar':
            print('Polar (phi,nu,r):  (%.2f, %.2f, %.2f)' % (self.phi, self.nu, self.r))


class PositionalSource(object):
    def __init__(self, cnp.ndarray[float, ndim=1] signal, position, int sample_rate):
        assert not isinstance(position, list)
        assert signal.ndim == 1
        self.signal = signal
        self.position = position
        self.sample_rate = sample_rate


class MovingSource(PositionalSource):
    def __init__(self, cnp.ndarray[float, ndim=1] signal, positions, int rate):
        super(MovingSource, self).__init__(signal, Position(0, 0, 0, 'polar'), rate)
        # PositionalSource.__init__(self, signal, Position(0, 0, 0, 'polar'), rate)

        cdef double duration = signal.shape[0] / float(rate)
        self.pts_p = positions
        self.npts = len(self.pts_p)
        self.pts_t = np.linspace(0, duration, self.npts)
        self.nframes = int(duration * rate)
        self.dt = 1/float(rate)

        self.pts_idx = np.floor(np.linspace(0, (self.npts-1), self.nframes)).astype(int)
        self.cur_idx = -1

    def tic(self):
        if self.cur_idx >= (self.nframes-1):
            return False

        self.cur_idx += 1
        cdef double cur_t = self.cur_idx * self.dt
        cdef int idx = self.pts_idx[self.cur_idx]
        if idx == (self.npts-1):
            self.position = self.pts_p[-1]
        else:
            alpha = (cur_t - self.pts_t[idx]) / (self.pts_t[idx+1] - self.pts_t[idx])
            cur_pos = alpha * self.pts_p[idx + 1].coords('polar') + (1 - alpha) * self.pts_p[idx].coords('polar')
            self.position.set_polar(cur_pos[0], cur_pos[1], cur_pos[2])
        return True

        
class AmbisonicArray(object):
    def __init__(self, cnp.ndarray data, ambi_format=AmbiFormat()):
        self.data = data
        self.format = ambi_format

    def convert(self, int sample_rate, str ordering, str normalization):
        assert sample_rate is not None or ordering is not None or normalization is not None
        cdef int n = self.format.num_channels

        if sample_rate is not None and sample_rate != self.format.sample_rate:
            duration = float(self.data.shape[0]) / self.format.sample_rate
            data = resample(self.data, int(duration * sample_rate))
            self.format.sample_rate = sample_rate
        else:
            data = np.copy(self.data)

        if ordering is not None and ordering != self.format.ordering:
            assert ordering in CHANNEL_ORDERING
            mapping = map(lambda x: convert_ordering(
                x, ordering, self.format.ordering), range(n))
            data = data[:, mapping]
            self.format.ordering = ordering

        if normalization is not None and normalization != self.format.normalization:
            assert normalization in NORMALIZATION
            c_out = np.array(map(lambda x: normalization_factor(
                x, self.format.ordering, normalization), range(n)))
            c_in = np.array(map(lambda x: normalization_factor(
                x, self.format.ordering, self.format.normalization), range(n)))
            data *= (c_out / c_in).reshape((1, -1))
            self.format.normalization = normalization

        self.data = data

class AmbiEncoder(object):
    def __init__(self, ambi_format=AmbiFormat()):
        self.format = ambi_format

    def encode(self, sources):
        if isinstance(sources, PositionalSource):
            sources = [sources]

        fmt = self.format
        Y = spherical_harmonics_matrix([src.position for src in sources], fmt.order)
        cdef cnp.ndarray[float, ndim=2] src_signals = np.stack([src.signal for src in sources], axis=1)
        return AmbisonicArray(np.dot(src_signals, Y), self.format)

    def encode_frame(self, sources, ambi_array, int frame_no):
        if isinstance(sources, PositionalSource):
            sources = [sources]

        cdef cnp.ndarray[float, ndim=2] src_signal = np.array([src[frame_no].signal for src in sources]).T
        cdef cnp.ndarray[float, ndim=2] Y = spherical_harmonics_matrix([src[frame_no].position for src in sources], self.format.order)
        
        frame_size = ambi_array.data[frame_no].shape[0]
        src_signal = src_signal[0:frame_size]
        ambi_array.data[frame_no] = np.dot(src_signal, Y)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def mono_to_foa_dynamic(cnp.ndarray[float, ndim=1] x, int n_frames, int sample_rate, int n_positions):
    """
    Uses the seld_ambisonics to spatialize mono files with dynamic moviment
    """

    fmt = AmbiFormat(ambi_order=1, sample_rate=sample_rate, ordering="ACN")
    encoder = AmbiEncoder(fmt)
    
    cdef int pt = 0
    cdef cnp.ndarray _X = np.arange(0, n_frames)

    windows = np.random.dirichlet(np.ones(n_positions), size=1)[0]
    windows = windows * n_frames

    x_positions = []
    positions1, positions2, positions3 = [], [], []
    pt = 0
    for i in range(n_positions):
        # Randomly select a value in the specified range
        coord1 = random.choice(PHI)
        coord2 = random.choice(ELE)
        coord3 = random.choice(Z)

        for _ in range(round(windows[i])):
            positions1.append(coord1)
            positions2.append(coord2)
            positions3.append(coord3)
            x_positions.append(pt)
            pt = pt + 1

    positions1 = gaussian_filter1d(positions1, sigma=3)
    positions2 = gaussian_filter1d(positions2, sigma=3) 
    positions3 = gaussian_filter1d(positions3, sigma=3)
    positions = list(zip(positions1, positions2, positions3))

    X_Y_Spline = make_interp_spline(x_positions, positions, k=2)
    positions = X_Y_Spline(_X)

    sources = []
    for (p, src) in zip(positions, np.array_split(x, n_frames)):
        sources.append(PositionalSource(src, Position(p[0], p[1], p[2], "polar"), sample_rate=sample_rate))
    
    assert len(sources) == n_frames
    
    cdef cnp.ndarray[float, ndim=3] ambi_array = np.zeros((n_frames, int(x.shape[0]/n_frames), fmt.num_channels), dtype = np.float32)
    
    ambi = AmbisonicArray(ambi_array, fmt)
    
    for idx in range(len(sources)):
        encoder.encode_frame([sources], ambi, idx)
    
    cdef cnp.ndarray[float, ndim=3] ambi_data = ambi.data

    return ambi_data.reshape(-1, 4)


def mono_to_foa_dynamic_overlap(waves, int n_frames, int sample_rate, int n_positions):
    """
    Uses the seld_ambisonics repo to spatialize mono files with dynamic
    """

    fmt = AmbiFormat(ambi_order=1, sample_rate=sample_rate, ordering="ACN")
    encoder = AmbiEncoder(fmt)
    
    cdef int pt = 0
    cdef cnp.ndarray _X = np.arange(0, n_frames)
    
    sources_waves = []
    for i in range(len(waves)):
    
        x = waves[i]

        windows = np.random.dirichlet(np.ones(n_positions), size=1)[0]
        windows = windows * n_frames

        x_positions = []
        positions1, positions2, positions3 = [], [], []
        pt = 0
        for i in range(n_positions):
            # Randomly select a value in the specified range
            coord1 = random.choice(PHI)
            coord2 = random.choice(ELE)
            coord3 = random.choice(Z)

            for _ in range(round(windows[i])):
                positions1.append(coord1)
                positions2.append(coord2)
                positions3.append(coord3)
                x_positions.append(pt)
                pt = pt + 1

        positions1 = gaussian_filter1d(positions1, sigma=3)
        positions2 = gaussian_filter1d(positions2, sigma=3) 
        positions3 = gaussian_filter1d(positions3, sigma=3)
        positions = list(zip(positions1, positions2, positions3))

        X_Y_Spline = make_interp_spline(x_positions, positions, k=2)
        positions = X_Y_Spline(_X)

        sources = []
        for (p, src) in zip(positions, np.array_split(x, n_frames)):
            sources.append(PositionalSource(src, Position(p[0], p[1], p[2], "polar"), sample_rate=sample_rate))
        
        assert len(sources) == n_frames
        
        sources_waves.append(sources)
        
    cdef cnp.ndarray[float, ndim=3] ambi_array = np.zeros((n_frames, int(x.shape[0]/n_frames), fmt.num_channels), dtype = np.float32)
    
    ambi = AmbisonicArray(ambi_array, fmt)
    
    for idx in range(n_frames):
        encoder.encode_frame(sources_waves, ambi, idx)
    
    cdef cnp.ndarray[float, ndim=3] ambi_data = ambi.data

    return ambi_data.reshape(-1, 4)