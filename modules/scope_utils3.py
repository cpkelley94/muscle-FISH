"""
scope_utils3.py            Chase Kelley             modified 03/01/2020
----------------------------------------------------------------------
A personal library of useful tools to visualize, analyze, and interpret \
microscopy imaging data.  Many functions in this library are simple \
to run either within a script or directly from the command line using \
`script.py`.
"""
import matplotlib
matplotlib.use('Agg')

from czifile import CziFile
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation, rcParams
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy import ndimage, optimize
from scipy.signal import argrelextrema
from skimage import filters, feature, measure, restoration, transform
from xml.etree import ElementTree
import cv2
import numpy as np
# import psf


# visualization convenience functions -----------------------------------------#

def animate_zstack(img, frames=None, title=None, vmin=None, vmax=None, gif_name=None, **kwargs):
    """
    Convenience function to plot and show a z-stack animation of any 3D image data.
    """
    def update_frame(f):
        imxy.set_data(img[:,:,f])
        return imxy

    if not frames:
        frames = list(range(img.shape[2]))
    if not vmin:
        vmin = np.amin(img)
    if not vmax:
        vmax = np.amax(img)

    fig, ax = plt.subplots()
    imxy = ax.imshow(img[:,:,frames[0]], vmin=vmin, vmax=vmax, **kwargs)
    if title:
        ax.set_title(title)
    anim = FuncAnimation(fig, update_frame, frames=frames, interval=200, blit=False)
    if gif_name:
        # Writer = animation.writers['imagemagick']
        # writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        anim.save(gif_name, writer='imagemagick', fps=8, dpi=300)
    # plt.show()
    return True

def animate_zstacks(img_list, frames=None, titles=None, vmin=None, vmax=None, cmaps=None, interval=200, gif_name=None, bgcolor=None, **kwargs):
    """
    Convenience function to plot a z-stack animation of a list of 3D images. \
    All images must have the same length in the z-dimension.
    """
    def update_frame(f):
        for i, img in enumerate(img_list):
            imxy[i].set_data(img[:,:,f])
        return imxy

    if not all([img.shape[2] == img_list[0].shape[2] for img in img_list]):
        print('Error: all images must have same length in z-dimension.')

    if not frames:
        frames = list(range(img_list[0].shape[2]))
    if not vmin:
        vmin = [np.amin(img) for img in img_list]
    if not vmax:
        vmax = [np.amax(img) for img in img_list]
    if not cmaps:
        cmaps = ['binary_r'] * len(img_list)

    fig, ax = plt.subplots(1, len(img_list), figsize=(3.*len(img_list), 3.))
    imxy = []
    for i, img in enumerate(img_list):
        if bgcolor:
            ax[i].set_facecolor(bgcolor)
        imxy.append(ax[i].imshow(img[:,:,frames[0]], vmin=vmin[i], vmax=vmax[i], cmap=cmaps[i], **kwargs))
    if titles:
        for i, title in enumerate(titles):
            ax[i].set_title(title)
    anim = FuncAnimation(fig, update_frame, frames=frames, interval=interval, blit=False)
    if gif_name:
        # Writer = animation.writers['imagemagick']
        # writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        anim.save(gif_name, writer='imagemagick', fps=8, dpi=300)
    # plt.show()
    plt.close()
    return True

def animate_zstack_from_czi(czi, gain=1., separate=False, **kwargs):
    """
    Animate the z-stack in a given CZI file.  Unless separate is True, plot \
    all channels as an overlay, with the legend constructed from CZI file \
    metadata.  If separate is True, plot all channels on separate subplots.

    Currently only works with Airyscan images.
    """
    def update_frame(f):
        for k, imxy in enumerate(imgs_xy):
            imxy.set_data(channels[k][:,:,f])
        return imxy

    # interpret `czi` argument
    if isinstance(czi, str):
        # interpret as path to CZI file
        with CziFile(czi) as czi_file:
            meta = czi_file.metadata()
            img = czi_file.asarray()
    elif isinstance(czi, CziFile):
        # interpret as CziFile object
        meta = czi.metadata()
        img = czi.asarray()
    else:
        # unknown argument type
        raise TypeError('`czi` argument must be of type `str` or `CziFile`.')

    gain = float(gain)

    # attempt to get channels and make legend
    mtree = ElementTree.fromstring(meta)
    track_tree = mtree.find('Metadata').find('Experiment').find('ExperimentBlocks').find('AcquisitionBlock').find('MultiTrackSetup').findall('TrackSetup')

    tracks = []
    for track in track_tree:
        name = track.get('Name')
        wavelen = float(track.find('Attenuators').find('Attenuator').find('Wavelength').text)*1E9
        rgb = np.array(wavelength_to_rgb(wavelen))/256.
        # print rgb
        tracks.append([name, rgb])

    channels = []
    for t in range(len(tracks)):
        channels.append(img[0, 0, t, 0, :, :, :, 0].transpose(1, 2, 0))  # x, y, z

    # make colormaps
    cmaps = []
    for j, track in enumerate(tracks):
        cmaps.append(LinearSegmentedColormap.from_list('cmap' + str(j), [list(tracks[j][1]) + [0.], list(tracks[j][1]) + [1.]]))

    frames = list(range(channels[0].shape[2]))
    vmins = [np.amin(channels[i]) for i in range(len(channels))]
    vmaxs = [np.amax(channels[i])/gain for i in range(len(channels))]

    if not separate:
        fig, ax = plt.subplots()
        ax.set_facecolor('k')
        imgs_xy = []
        for i in range(len(channels)):
            imgs_xy.append(ax.imshow(channels[i][:,:,frames[0]], vmin=vmins[i], vmax=vmaxs[i], cmap=cmaps[i], **kwargs))
        # ax.legend([l[0] for l in tracks])
    else:
        fig, ax = plt.subplots(1, len(channels))
        imgs_xy = []
        for i in range(len(channels)):
            ax[i].set_facecolor('k')
            imgs_xy.append(ax[i].imshow(channels[i][:,:,frames[0]], vmin=vmins[i], vmax=vmaxs[i], cmap=cmaps[i], **kwargs))
            ax[i].set_title(tracks[i][0])

    anim = FuncAnimation(fig, update_frame, frames=frames, interval=200, blit=False)
    # plt.show()

    return True


# 2D image analysis -----------------------------------------------------------#

def count_cells(image, band_min=10., band_max=30., r_min=np.sqrt(2.), r_max=10.*np.sqrt(2.), threshold=0.095, draw=False):
    """
    Count the number of cells in a 10X phase contrast microscopy image. Performs \
    a 2D Fourier transform and applies a soft (Gaussian) bandpass filter to enrich \
    frequencies corresponding to cell radii between `band_min` and `band_max`. \
    Detects cells in the filtered image using the Laplacian of Gaussian approach \
    implemented in `scikit_image.feature.blob_log`. Returns the number of cells \
    counted in the image.

    For blob detection, minimum particle radius `r_min`, maximum particle radius \
    `r_max`, and intensity threshold should be optimized for cell type and \
    microscope parameters. Defaults were chosen empirically using 10X phase \
    contrast images of an undifferentiated Neuro2a cell line taken on the EVOS FL \
    (58% light intensity).

    Requires a 2D grayscale image as a numpy array.
    """

    # discrete Fourier transform
    f = np.fft.fft2(image)
    x = np.fft.fftfreq(f.shape[1])
    y = np.fft.fftfreq(f.shape[0])

    # generate mask for bandpass filtering
    X, Y = np.meshgrid(x, y)
    zero_pass = kronecker_delta(X, Y, 0., 0.)  # preserve baseline intensity at stationary points
    gauss_max_pos = gaussian(X, Y, 0., 0., 0, 0.5/(np.sqrt(2.)*band_min))  # low-pass filter boundary
    gauss_min_pos = mesh_zeros(X, Y) - gaussian(X, Y, 0., 0., 0, 0.5/(np.sqrt(2.)*band_max))  # high-pass filter boundary
    bp_filter = gauss_max_pos + gauss_min_pos + zero_pass

    # apply mask and transform back to space domain
    f_bp = np.multiply(f, bp_filter)
    image_bp = normalize_image(np.abs(np.fft.ifft2(f_bp)))

    # detect cells
    blobs = feature.blob_log(normalize_image(image_bp, 1, 0), min_sigma=r_min/np.sqrt(2.), max_sigma=r_max/np.sqrt(2.), num_sigma=20, threshold=threshold)

    if draw:
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='Greys_r')
        ax.scatter([r[1] for r in blobs], [r[0] for r in blobs], marker='+', color='r')
        # plt.show()

    return len(blobs)

def detect_local_minima(arr):
    # https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
    """
    Takes an array and detects the troughs using the local maximum filter.
    Returns a boolean mask of the troughs (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """
    # define an connected neighborhood
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#generate_binary_structure
    neighborhood = ndimage.morphology.generate_binary_structure(len(arr.shape),2)
    # apply the local minimum filter; all locations of minimum value
    # in their neighborhood are set to 1
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.filters.html#minimum_filter
    local_min = (ndimage.filters.minimum_filter(arr, footprint=neighborhood)==arr)
    # local_min is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.
    #
    # we create the mask of the background
    background = (arr==0)
    #
    # a little technicality: we must erode the background in order to
    # successfully subtract it from local_min, otherwise a line will
    # appear along the background border (artifact of the local minimum filter)
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#binary_erosion
    eroded_background = ndimage.morphology.binary_erosion(background, structure=neighborhood, border_value=1)
    #
    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_min mask
    detected_minima = local_min ^ eroded_background
    return np.where(detected_minima)

def find_cells(image, band_min=10., band_max=30., r_min=np.sqrt(2.), r_max=10.*np.sqrt(2.), threshold=0.095, draw=False):
    """
    Find positions of cells in a 10X phase contrast microscopy image. Performs \
    a 2D Fourier transform and applies a soft (Gaussian) bandpass filter to enrich \
    frequencies corresponding to cell radii between `band_min` and `band_max`. \
    Detects cells in the filtered image using the Laplacian of Gaussian approach \
    implemented in `scikit_image.feature.blob_log`. Returns a list of blob objects \
    returned by `blob_log`.

    For blob detection, minimum particle radius `r_min`, maximum particle radius \
    `r_max`, and intensity threshold should be optimized for cell type and \
    microscope parameters. Defaults were chosen empirically using 10X phase \
    contrast images of an undifferentiated Neuro2a cell line taken on the EVOS FL \
    (58% light intensity).

    Requires a 2D grayscale image as a numpy array.
    """

    # discrete Fourier transform
    f = np.fft.fft2(image)
    x = np.fft.fftfreq(f.shape[1])
    y = np.fft.fftfreq(f.shape[0])

    # generate mask for bandpass filtering
    X, Y = np.meshgrid(x, y)
    zero_pass = kronecker_delta(X, Y, 0., 0.)  # preserve baseline intensity at stationary points
    gauss_max_pos = gaussian(X, Y, 0., 0., 0, 0.5/(np.sqrt(2.)*band_min))  # low-pass filter boundary
    gauss_min_pos = mesh_zeros(X, Y) - gaussian(X, Y, 0., 0., 0, 0.5/(np.sqrt(2.)*band_max))  # high-pass filter boundary
    bp_filter = gauss_max_pos + gauss_min_pos + zero_pass

    # apply mask and transform back to space domain
    f_bp = np.multiply(f, bp_filter)
    image_bp = normalize_image(np.abs(np.fft.ifft2(f_bp)))

    # detect cells
    blobs = feature.blob_log(normalize_image(image_bp, 1, 0), min_sigma=r_min/np.sqrt(2.), max_sigma=r_max/np.sqrt(2.), num_sigma=20, threshold=threshold)

    if draw:
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='Greys_r')
        ax.scatter([r[1] for r in blobs], [r[0] for r in blobs], marker='+', color='r')
        # plt.show()

    return blobs

# 3D image analysis -----------------------------------------------------------#

def autofocus_zstack(image, plot_opt=False):
    """
    This function searches the z-stack for the z-plane that is most in focus. \
    For each z-plane, the function calculates the 2-D discrete Laplacian by \
    convolution and calculates its variance.  The function returns the index of \
    the z-plane that maximizes the variance of the Laplacian normalized by the \
    square of the mean pixel intensity.

    Dependencies: cv2, numpy
    """
    # note: still unsure why scipy.ndimage.laplace() and cv2.Laplacian() return completely different results
    var_lap_over_mean_sq = [cv2.Laplacian(image[:, :, z], cv2.CV_64F).var()/(np.mean(image[:, :, z])**2.) for z in range(image.shape[2])]

    if plot_opt:
        plt.plot(list(range(image.shape[2])), var_lap_over_mean_sq, 'k-', linewidth=0.8)
        # plt.show()

    print('Optimized z-plane for autofocus: ' + str(np.argmax(var_lap_over_mean_sq)))

    return np.argmax(var_lap_over_mean_sq)


def deconvolve_widefield(img, metadata, ex_wavelen, em_wavelen, psfsize_xy=50, psfsize_z=25, bit_depth=16, plot_psf=False, **kwargs):
    """
    NOTE: This approach doesn't seem to be effective, as the PSF of the Zeiss \
    widefield microscope is poorly modeled by this z-symmetric function.

    Deconvolve a widefield fluorescence microscopy image in CZI format using the \
    Richardson-Lucy algorithm. Extracts parameters from the CZI file metadata \
    and generates a point spread function using Christoph Gohlke's \'psf\' \
    module.

    Dependencies: scikit-image, psf, numpy

    Arguments:
        img (ndarray) :  A 3D numpy array arranged as [x, y, z].
        metadata (str) :  Zeiss file metadata, extracted from the CziFile object using czi.metadata().
        ex_wavelen (float) :  Excitation wavelength, in nanometers.
        em_wavelen (float) :  Emission wavelength, in nanometers.
        psfsize_xy (int) :  The pixel size of the point spread function in the x- and y- dimensions.
        psfsize_z (int) :  The pixel size of the point spread function in the z- dimension.
        bit_depth (int) :  Bit depth of the image, in number of bits.  Default 16.
        plot_psf (bool) : Flag to plot a slice of the point spread function. Requires matplotlib. Default False.

        kwargs are passed to psf.PSF() and will override parameters parsed from \
        the CZI metadata.

    Returns:
        deconv (ndarray) :  Deconvolved image, same dimensions as \'img\'.
    """

    # extract important PSF parameters
    mtree = ElementTree.fromstring(metadata)
    scales = mtree.find('Metadata').find('Scaling').find('Items').findall('Distance')
    for scale in scales:
        if scale.get('Id') == 'X':
            xy_scale = float(scale.find('Value').text)*1.E6
        elif scale.get('Id') == 'Z':
            z_scale = float(scale.find('Value').text)*1.E6

    try:
        xy_scale
    except NameError:
        print('Error: xy-scale not found in CZI metadata.')
        return False
    try:
        z_scale
    except NameError:
        print('Error: z-scale not found in CZI metadata.')
        return False

    num_aperture = float(mtree.find('Metadata').find('Information').find('Instrument').find('Objectives').find('Objective').find('LensNA').text)
    refr_index = float(mtree.find('Metadata').find('Information').find('Image').find('ObjectiveSettings').find('RefractiveIndex').text)
    magnification = float(mtree.find('Metadata').find('Information').find('Instrument').find('Objectives').find('Objective').find('NominalMagnification').text)

    params = {'name':'zeiss',
              'shape':(psfsize_z, psfsize_xy),
              'dims':(z_scale*psfsize_z, xy_scale*psfsize_xy),
              'ex_wavelen':ex_wavelen,
              'em_wavelen':em_wavelen,
              'num_aperture':num_aperture,
              'refr_index':refr_index,
              'magnification':magnification}

    # overrides and kwargs
    if kwargs:
        for key, val in kwargs.items():
            params.update({key:val})

    # generate point spread function
    psf_object = psf.PSF(psf.ISOTROPIC | psf.WIDEFIELD, **params)
    psf_vol = psf_object.volume().transpose((1,2,0))

    if plot_psf:
        psf_slice = psf_vol[int(np.floor(psf_vol.shape[0]/2)), :, :].transpose((1,0))
        plt.imshow(np.log10(psf_slice), cmap='inferno', interpolation='lanczos')
        # plt.show()

    # deconvolve using the Richardson-Lucy algorithm
    deconv = restoration.richardson_lucy(normalize_image(img, vmax=np.amax(img)/(2.**bit_depth)), psf_vol)

    return deconv

def normalize_image(image, vmin=0, vmax=1):
    """
    Convenience function to normalize a numpy array to given bounds.  Default 0 \
    to 1.
    """
    return np.interp(image, (np.min(image), np.max(image)), (vmin, vmax))

def optimize_threshold(image_log, plot_opt=False):
    """
    Find the optimal threshold for FISH spot detection in the Laplacian of \
    Gaussian filtered image.  This is accomplished by testing many possible \
    threshold values, identifying the number of detected spots for each, \
    and choosing the threshold that corresponds to the inflection point of the \
    plateau.  This technique is described in more detail in various publications \
    by the Arjun Raj lab.  Return the optimal threshold value.
    """
    # find the threshold range
    t_max = np.max(image_log)
    t_min = np.min(image_log)
    t_range = np.linspace(t_min, t_max, num=100)

    # for each threshold, count the number of spots detected
    num_spots = []
    for t in t_range:
        image_bin = np.where(image_log < t, 1, 0)
        labels, n = ndimage.label(image_bin)
        num_spots.append(n)

    if plot_opt:
        plt.plot(t_range, num_spots, 'k-', linewidth=0.8)
        # plt.show()

    dn_dt = np.gradient(num_spots, t_range[1]-t_range[0])

    if plot_opt:
        plt.plot(t_range, dn_dt, 'b-', linewidth=0.8)
        # plt.show()

    # find optimal threshold
    inflection_pts = argrelextrema(dn_dt, np.less)[0]
    spike = np.argmax(num_spots)
    t_opt = t_range[[pt for pt in inflection_pts if pt < spike][-1]]
    print('Optimal threshold value for FISH LoG: ' + str(t_opt))

    return t_opt

def optimize_threshold_blob_log(image, min_sigma=1., max_sigma=50., num_sigma=10, tmin=0., tmax=0.1, num=20, plot_opt=False):
    """
    Find the optimal threshold for FISH spot detection in the Laplacian of \
    Gaussian filtered image.  This is accomplished by testing many possible \
    threshold values, identifying the number of detected spots for each, \
    and choosing the threshold that corresponds to the inflection point of the \
    plateau.  This technique is described in more detail in various publications \
    by the Arjun Raj lab.  Return the optimal threshold value.
    """
    # find the threshold range
    t_range = np.linspace(tmin, tmax, num=num)

    # for each threshold, count the number of spots detected
    num_spots = []
    for t in t_range:
        blobs = feature.blob_log(image, min_sigma, max_sigma, num_sigma, threshold=t)
        num_spots.append(len(blobs))

    if plot_opt:
        plt.plot(t_range, num_spots, 'k-', linewidth=0.8)
        # plt.show()

    dn_dt = np.gradient(num_spots, t_range[1]-t_range[0])

    if plot_opt:
        plt.plot(t_range, dn_dt, 'b-', linewidth=0.8)
        # plt.show()

    # find optimal threshold
    inflection_pts = argrelextrema(dn_dt, np.less)[0]
    spike = np.argmax(num_spots)
    t_opt = t_range[[pt for pt in inflection_pts if pt < spike][-1]]
    print('Optimal threshold value for blob_log: ' + str(t_opt))

    return t_opt

# mathematical constructions and optimization functions -----------------------#

def gauss_1d(x, A, mu, sig):
    return A*np.exp(-1.*((x-mu)**2.)/(2.*(sig**2.)))

def gauss_2d(x, y, x_off, y_off, mu, sigma):
    d = np.sqrt((x-x_off)**2. + (y-y_off)**2.)
    return np.exp(-1.*(d**2.)/(2.*(sigma**2.)))

def gauss_3d(pos, A, mu_x, mu_y, mu_z, sig_xy, sig_z, m_x, m_y, m_z, b):
    '''
    Model to fit to FISH spots using scipy.optimize.curve_fit().  Spots are \
    modeled using a 3D Gaussian function over a linear baseline.
    '''
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    return A*np.exp(-1.*((((x-mu_x)**2.)/(2.*(sig_xy**2.)))+(((y-mu_y)**2.)/(2.*(sig_xy**2.)))+(((z-mu_z)**2.)/(2.*(sig_z**2.))))) + m_x*x + m_y*y + m_z*z + b

def kronecker_delta(x, y, x_off, y_off):
    if type(x) != np.ndarray and type(y) != np.ndarray:
        if x == x_off and y == y_off:
            return 1.
        else:
            return 0.
    else:
        x_off_mat = x_off*np.ones(x.shape)
        y_off_mat = y_off*np.ones(y.shape)
        return np.logical_and(np.equal(x, x_off_mat), np.equal(y, y_off_mat), dtype=int)

def mesh_zeros(x, y):
    return 0.

# miscellaneous functions -----------------------------------------------------#

def get_czi_metadata(p_czi, p_out=None, stdout=False):
    """
    Convenience function to strip the metadata out of a CZI image and return. \
    Requires CZI path.  If `p_out` is provided, log the output to file.  If \
    `stdout` is True, print the metadata to standard output.
    """
    with CziFile(p_czi) as czi:
        meta = czi.metadata()
    if p_out:
        with open(p_out, 'w') as outfile:
            outfile.writelines(meta)
    if stdout:
        print(meta)
    return meta

def wavelength_to_rgb(wavelength, gamma=0.8):
    '''This converts a given wavelength of light to an
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    '''

    wavelength = float(wavelength)
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    R *= 255
    G *= 255
    B *= 255
    return (int(R), int(G), int(B))

# define colormaps
cmap_KtoR = LinearSegmentedColormap.from_list('cmap_KtoR', ['#000000', '#ff0000'])
cmap_KtoG = LinearSegmentedColormap.from_list('cmap_KtoG', ['#000000', '#00ff00'])
cmap_KtoB = LinearSegmentedColormap.from_list('cmap_KtoB', ['#000000', '#2255ff'])
cmap_NtoR = LinearSegmentedColormap.from_list('cmap_NtoR', ['#ff000000', '#ff0000ff'])
cmap_NtoG = LinearSegmentedColormap.from_list('cmap_NtoG', ['#00ff0000', '#00ff00ff'])
cmap_NtoB = LinearSegmentedColormap.from_list('cmap_NtoB', ['#2255ff00', '#2255ffff'])
cmap_NtoW = LinearSegmentedColormap.from_list('cmap_NtoW', ['#ffffff00', '#ffffffff'])
