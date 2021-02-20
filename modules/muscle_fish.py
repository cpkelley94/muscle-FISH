# python 3.6.5, HiPerGator
"""
Library of functions for analysis of muscle fiber HCR-FISH images.

Note: in general, these function assume 3D image arrays are organized as 
(x, y, z). If you prefer images in the order (z, x, y), use the `z_first` 
optional argument.
"""

from itertools import count
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import optimize
from skimage import feature, exposure, filters, morphology
from xml.etree import ElementTree
import numpy as np
import os
import scope_utils3 as su


def open_image(img_path, meta_path=None, z_first=False):
    """Open a multichannel 3D microscopy image.

    Open a multichannel 3D microscopy image and return the image as a 4-D numpy 
    array. Accepted filetypes are CZI image (`.czi`) or OME-TIFF image 
    (`.ome.tiff` or `.ome.tif`). Also return the metadata as an XML 
    ElementTree object, either scraped from the CZI image or grabbed from the 
    XML file provided at `meta_path`. By default, the image is returned with 
    the dimensions [c, x, y, z].

    For CZI images, the `czifile` package is required. For OME-TIFF images, the 
    `bioformats` and `javabridge` packages are required, as well as Java.

    Parameters:

    img_path: str
        Path to the microscopy image. CZI and OME-TIFF files are accepted.


    Optional:

    meta_path: str (default None)
        Path to the metadata XML file. If a `img_path` points to a CZI file, 
        this argument is not required, but if provided, it will be used over 
        the CZI metadata. If `img_path` points to an OME-TIFF file, `meta_path` 
        will be used to read the image's metadata. If `meta_path` is not 
        provided, search the image directory for an XML file with the same 
        basename as `img_path`.

    z_first: bool (default False)
        If True, return `img` with the dimensions [c, z, x, y].
    """

    # determine image filetype and open
    if img_path.lower().endswith('.czi'):
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        # open CZI format using `czifile` library
        from czifile import CziFile
        with CziFile(img_path) as czi_file:
            img_czi = czi_file.asarray()
            if meta_path is not None:
                metatree = ElementTree.parse(meta_path)
            else:
                meta = czi_file.metadata()
                metatree = ElementTree.fromstring(meta)
        
        if z_first:
            img = img_czi[0,0,:,0,:,:,:,0]  # c, z, x, y
        else:
            img = img_czi[0,0,:,0,:,:,:,0].transpose(0,2,3,1)  # c, x, y, z

    elif any([img_path.lower().endswith(ext) for ext in ['.ome.tiff', '.ome.tif']]):
        img_name = os.path.splitext(os.path.splitext(os.path.basename(img_path))[0])[0]

        # open OME-TIFF format using `bioformats` library (requires Java)
        import javabridge
        import bioformats
        javabridge.start_vm(class_path=bioformats.JARS)

        # iterate over z-stack until end of file
        slices = []
        for z in count():
            try:
                s = bioformats.load_image(img_path, z=z)
                slices.append(s)
            except javabridge.jutil.JavaException:
                # final z-slice was read, stop reading
                javabridge.kill_vm()
                break

        img_ome = np.stack(slices)

        if z_first:
            img = img_ome.transpose(3,0,1,2)  # c, z, x, y
        else:
            img = img_ome.transpose(3,1,2,0)  # c, x, y, z

        # look for metadata .XML file with same filename
        if meta_path is None:
            meta_path = os.path.splitext(os.path.splitext(img_path)[0])[0] + '.xml'
        try:
            metatree = ElementTree.parse(meta_path)
        except IOError:
            # metadata file not found
            raise IOError('CZI metadata XML not found at expected path "' + meta_path + '" (required for OME-TIFF)')

    else:
        raise ValueError('Image filetype not recognized. Allowed:  .CZI, .OME.TIFF')

    return img, img_name, metatree


def threshold_nuclei(img_dapi, t_dapi=None, fiber_mask=None, labeled=True, multiplier=0.75, z_first=False, verbose=False):
    """Segment nuclei in a 3D DAPI image.

    Return a mask containing all nuclei from a 3D DAPI image. If `fiber_mask` 
    is provided, exclude any nuclei that do not intersect the fiber. If 
    `t_fiber` is provided, use that for thresholding; otherwise, determine the 
    threshold value automatically using Otsu's method.

    Parameters:

    img_dapi: np.array, 3D
        A 3D numpy array containing signal from the DAPI channel.

    
    Optional:

    t_fiber: float (default None)
        Threshold to use for binarization of the DAPI image, from 0 to 1. 
        If not set, automatically determine this threshold using Otsu's 
        method. 

    fiber_mask: np.array, 3D (default None)
        A 3D numpy array (`bool` or `int` type) containing a truth mask of the 
        muscle fiber. Used to exclude nuclei outside the fiber.
    
    labeled: bool (default True)
        If True, return a mask with each nucleus labeled as an integer and the 
        background labeled as 0. Otherwise, return a mask with all nuclear 
        pixels labeled as 1 and background as 0.
    
    multiplier: float (default 0.75)
        If using automatic thresholding of DAPI by Otsu's method, multiply the 
        threshold by this value before binarizing the DAPI channel.

    z_first: bool (default False)
        If True, treat `img_dapi` as image with dimensions [z, x, y].

    verbose: bool (default False)
        If True, print progress to stdout.
        
    """

    if verbose:
        print('\nSegmenting nuclei...')

    # image preprocessing
    if z_first:
        sig = (2, 20, 20)
    else:
        sig = (20, 20, 2)

    img_dapi = su.normalize_image(img_dapi)
    img_blur = filters.gaussian(img_dapi, sig)

    # get threshold using method determined from value of t_dapi
    if t_dapi is None:
        thresh = filters.threshold_otsu(img_blur)*multiplier
    elif type(t_dapi) == float:
        thresh = t_dapi
    else:
        raise TypeError('`t_dapi` argument not recognized. \
            Must be either float or None.')
    
    if verbose:
        print('DAPI threshold = ' + str(thresh))
    
    # binarize and clean up mask
    bin_dapi = np.where(img_dapi > thresh, 1, 0)
    bin_dapi = morphology.remove_small_objects(bin_dapi.astype(bool), 2048)
    bin_dapi = morphology.remove_small_holes(bin_dapi.astype(bool), 2048)
    nuclei_labeled, n_nuc = morphology.label(bin_dapi, return_num=True)

    if fiber_mask is not None:
        if verbose:
            print('Removing nuclei that are not connected to main fiber segment...')

        for i in range(1, n_nuc+1):
            overlap = np.logical_and(np.where(nuclei_labeled == i, True, False), fiber_mask.astype(bool))
            if np.count_nonzero(overlap) == 0:
                nuclei_labeled[nuclei_labeled == i] = 0

    if labeled:
        return nuclei_labeled
    else:
        return np.where(nuclei_labeled > 0, 1, 0)


def threshold_fiber(img_fish, t_fiber=None, z_first=False, verbose=False):
    """Segment the muscle fiber from a 3D FISH microscopy image.

    Return a mask containing the extent of the muscle fiber from a 3D FISH 
    image. If `t_fiber` is provided, use that for thresholding. Otherwise, 
    automatically determine the threshold value using Li's method.

    Parameters:

    img_fish: np.array, 3D
        A 3D numpy array containing signal from the FISH channel, organized 
        as [x, y, z] (unless `z_first` is set to True).


    Optional:
    
    t_fiber: float or str (default None)
        Threshold to use for binarization of the FISH image, from 0 to 1. 
        If not set, automatically determine this threshold using Li's method. 
        Alternatively, if set to 'last', the final z-slice of the image will 
        be thresholded, and this mask will be used for the entire z-stack.
    
    z_first: bool (default False)
        If True, treat `img_fish` as image with dimensions [z, x, y].

    verbose: bool (default False)
        If True, print progress to stdout.
    """

    if verbose:
        print('Segmenting fiber...')

    # image preprocessing
    if z_first:
        # switch to (x, y, z)
        img_fish = img_fish.transpose(1,2,0)

    img_avg = su.normalize_image(img_fish)
    img_blur = filters.gaussian(img_avg, (10,10,1))

    # get threshold using method determined from value of t_fiber
    if t_fiber is None:
        thresh = filters.threshold_li(img_blur)
    elif type(t_fiber) == float:
        thresh = t_fiber
    elif t_fiber == 'last':
        thresh = filters.threshold_li(img_blur[:,:,-1])
    else:
        raise TypeError('`t_fiber` argument not recognized. \
            Must be either float or "last".')
        
    if verbose:
        print('Fiber threshold: ' + str(thresh))
    
    # binarize
    bin_fiber = np.where(img_blur > thresh, 1, 0)

    if t_fiber == 'last':
        # only use the last frame to define the fiber volume
        # this is a last resort option to deal with glare in the RNA channel
        # warning: densities will be underestimated
        for z in range(bin_fiber.shape[2]):
            bin_fiber[:,:,z] = bin_fiber[:,:,-1]

        if verbose:
            print('OVERRIDE: using final frame mask for entire z-stack.')
    
    # keep only largest object in the image
    fiber_labeled, n_labels = morphology.label(bin_fiber, return_num=True)
    label_sizes = {i:np.count_nonzero(np.where(fiber_labeled == i, 1, 0)) for i in range(1, n_labels+1)}
    fiber_idx = max(label_sizes, key=label_sizes.get)
    mask = np.where(fiber_labeled == fiber_idx, 1, 0)

    if z_first:
        # switch back to (z, x, y)
        mask = mask.transpose(2,0,1)

    return mask


def find_spots_snrfilter(img_fish, snr=None, sigma=1., t_spot=0.025, mask=None, z_first=False, draw=False, imgprefix='fiber'):
    """Find FISH spots in a 3D microscopy image, filtering out low intensity spots.

    Find HCR-FISH spots in the FISH channel using `skimage.feature.blob_log`, 
    and filter out low-intensity spots using an automatic signal-to-noise 
    threshold filter.

    Parameters:

    img_fish: np.array, 3D
        A 3D numpy array containing signal from the FISH channel.
    

    Optional:

    snr: float or None (default None)
        The threshold value for signal-to-noise used for filtering out called 
        spots that have low intensity relative to the fiber background. If a 
        float is provided, that value will be used. Otherwise, the algorithm 
        will automatically determine the optimal threshold.

    sigma: float (default 1.0)
        The standard deviation of the Gaussian kernel used for blurring before 
        the Laplacian operation. Increasing this value will preferentially 
        detect larger spots.
    
    t_spot: float (default None)
        Threshold to use for spot detection by skimage.filters.blob_log, from 0 
        to 1. Typical values for this parameter fall between 0.01 and 0.05.

    mask: np.array, 3D (same dimensions as `img_fish`)
        If provided, spots detected outside the mask will not be returned.

    z_first: bool (default False)
        If True, treat `img_fish` as image with dimensions [z, x, y].

    draw: bool (default False)
        If True, plot a histogram of pixel intensities within the fiber.

    imgprefix: str (default 'fiber')
        If draw is True, use this string as the prefix for the plot filename.
    """

    if z_first:
        # transpose image and mask
        img_fish = su.normalize_image(img_fish.transpose(1,2,0))
        if mask is not None:
            mask = mask.transpose(1,2,0)
    else:
        img_fish = su.normalize_image(img_fish)


    spots = feature.blob_log(su.normalize_image(img_fish), sigma, sigma, num_sigma=1, threshold=t_spot)
    
    if mask is not None:
        spots_masked = []
        for spot in spots:
            spot_pos = tuple(spot[0:3].astype(int))
            if mask[spot_pos]:
                spots_masked.append(spot)
        spots_masked = np.row_stack(spots_masked)
    else:
        spots_masked = spots

    # get fiber background intensity for noise comparison
    if draw:
        fig, ax = plt.subplots()
        ax.hist(np.ma.masked_where(np.logical_not(mask), img_fish), bins=100, density=True, histtype='step')
        ax.set_xlabel('Normed intensity of fiber pixel')
        ax.set_ylabel('Probability density')
        plt.tight_layout()
        plt.savefig(imgprefix + '_intensity_hist.png', dpi=300)
        plt.close()

    fiber_intensities = np.ma.masked_where(np.logical_not(mask), img_fish).compressed()
    baseline = np.percentile(fiber_intensities, 25)
    img_blur = filters.gaussian(img_fish, (3,3,0.3))

    # automatically determine SNR threshold
    if snr is None:
        spot_intensities = []
        for spot in spots_masked:
            try:
                pos = tuple(spot[0:3].astype(int))
                spot_intensities.append(img_blur[pos])
            except IndexError:
                # spot coordinate not found, likely right on the edge of the image
                # skip this spot
                pass
        
        thresh = np.percentile(spot_intensities, 90)/(baseline*np.sqrt(10))

        snr_min = np.sqrt(10)
        if thresh > snr_min:
            snr = thresh
        else:
            snr = snr_min
    
    print('SNR threshold = ' + str(snr))

    # filter out low intensity spots
    spots_filtered = []
    spot_data = [['spot_id', 'x', 'y', 'z', 'intensity', 'SNR']]
    for c, spot in enumerate(spots_masked):
        x_int, y_int, z_int = tuple(spot[0:3].astype(int))

        try:
            intensity = img_blur[x_int, y_int, z_int]

            if intensity / baseline > snr:
                if z_first:
                    # switch back to correct coordinates
                    spots_filtered.append(spot[[2,0,1,3]])
                else:
                    spots_filtered.append(spot)

            spot_data.append([c] + list(spot[0:3]) + [intensity, intensity/baseline])

        except IndexError:
            print('WARNING: spot coordinates outside of image. Moving to next spot...')

    return np.row_stack(spots_filtered), spot_data


def fix_bleaching(img, second_half=True, mask=None, z_first=False, draw=False, imgprefix='fiber'):
    """Correct an image for photobleaching that occurs along the z-dimension.

    Correct `img` for photobleaching that occurs along the z-dimension using 
    an exponential curve fit approach. Return a corrected version of `img`. This 
    function accepts normalized and non-normalized images, although the curve 
    fit returns slightly different results due to differences in baseline 
    intensity.

    Parameters:

    img: np.array, 3D
        A 3D numpy array containing signal from a microscopy image channel.
    

    Optional:

    second_half: bool (default True)
        If True, only use the second half of the z-stack for exponential curve 
        fitting. Set this to True if the first z-slices in the stack are outside 
        or at the edge of the fiber. If False, use the entire z-stack for the 
        fit.

    mask: np.array, 3D (same dimensions as `img_fish`)
        If provided, spots detected outside the mask will not be returned.

    z_first: bool (default False)
        If True, treat `img_fish` as image with dimensions [z, x, y].

    draw: bool (default False)
        If True, plot a histogram of pixel intensities within the fiber.

    imgprefix: str (default 'fiber')
        If draw is True, use this string as the prefix for the plot filename.
    """

    def expo_fit(x, A, k):
        # exponential model for photobleaching
        return A*np.exp(k*x)

    if z_first:
        img = img.transpose(1,2,0)
        if mask is not None:
            mask = mask.transpose(1,2,0)

    # define which frames are used for the fit
    if second_half:
        start_frame = img.shape[2] // 2
    else:
        start_frame = 0

    # get the mean intensity for each z-slice
    if mask is not None:
        masked_img = np.ma.masked_where(np.logical_not(mask), img)
        intensities = [np.mean(masked_img[:,:,z].compressed()) for z in range(start_frame, img.shape[2])]
    else:
        intensities = [np.mean(img[:,:,z]) for z in range(start_frame, img.shape[2])]

    # fit exponential model to the mean intensities
    try:
        e_params, e_cov = optimize.curve_fit(expo_fit, 
            list(range(start_frame, img.shape[2])), intensities,
            p0=[np.mean(img), -0.1])
    except RuntimeError:
        # optimization failed
        print('WARNING: Bleaching correction optimization failed. Continuing without correction...')
        
        if z_first:
            return img.transpose(2,0,1)
        else:
            return img

    if draw:
        x_fit = np.linspace(0, img.shape[2], num=200)
        y_fit = expo_fit(x_fit, *e_params)

        fig, ax = plt.subplots()

        if mask is not None:
            all_intens = [np.mean(masked_img[:,:,z].compressed()) for z in range(img.shape[2])]
        else:
            all_intens = [np.mean(img[:,:,z]) for z in range(img.shape[2])]

        ax.plot(list(range(img.shape[2])), all_intens, 'k.')
        ax.plot(x_fit, y_fit, 'k-', lw=0.8)
        ax.set_xlabel('z slice')
        ax.set_ylabel('mean intensity over image')
        plt.tight_layout()
        plt.savefig(imgprefix+'_bleaching_correction.png', dpi=300)
        plt.close()

    if e_params[1] > 0:
        # bad fit, do not correct
        print('WARNING: Bleaching correction optimization failed. Continuing without correction...')
        
        if z_first:
            return img.transpose(2,0,1)
        else:
            return img

    # correct img using the fit function inverse
    correction = np.array([expo_fit(z, 1, -1.*e_params[1]) for z in range(img.shape[2])])
    corrected_img = img*correction

    if z_first:
        return corrected_img.transpose(2,0,1)
    else:
        return corrected_img


# experimental ----------------------------------------------------------------#

def find_spots_gaussfit(img_fish, snr=4., sigma=1., t_spot=None, mask=None, draw=False, imgprefix='fiber'):
    """
    Find spots using LoG and filter out low-intensity spots. 

    Has comparable results to find_spots_snrfilter() but is orders of magnitude 
    slower.  Use find_spots_snrfilter() instead.
    """

    def update_z_spot(fm):
        # update fish frame
        im.set_data(img_fish[x_int-10:x_int+10, y_int-10:y_int+10, fm])
        if plot_circ:
            alpha = np.exp(-1.*((float(fm)-z)**2.)/(2.*(sig_z**2.)))
            circ.set_edgecolor((1., 0., 0., alpha))
            return im, circ

        return im

    spots = feature.blob_log(su.normalize_image(img_fish), sigma, sigma, num_sigma=1, threshold=t_spot)
    
    if mask is not None:
        spots_masked = []
        for spot in spots:
            spot_pos = tuple(spot[0:3].astype(int))
            if mask[spot_pos]:
                spots_masked.append(spot)
        spots_masked = np.row_stack(spots_masked)
    else:
        spots_masked = spots

    # get fiber background intensity for noise comparison
    fiber_intensities = np.ma.masked_where(np.logical_not(mask), img_fish).compressed()

    if draw:
        fig, ax = plt.subplots()
        ax.hist(fiber_intensities, bins=100, density=True, histtype='step')
        ax.set_xlabel('Normed intensity of fiber pixel')
        ax.set_ylabel('Probability density')
        plt.tight_layout()
        plt.savefig(imgprefix + '_intensity_hist.png', dpi=300)
        plt.close()

    baseline = np.percentile(fiber_intensities, 25)
    print(baseline)

    # at each spot position, fit a 3D Gaussian and get intensity
    spots_filtered = []
    spot_data = [['spot_id', 'x', 'y', 'z', 'A', 'SNR']]
    for c, spot in enumerate(spots_masked):
        # find an approximate z value for the spot as a guess for the fit
        x_int, y_int, z_int = tuple(spot[0:3].astype(int))
        
        # if draw:
        #     fig_im, ax_im = plt.subplots()
        #     im = ax_im.imshow(img_fish[x_int-10:x_int+10, y_int-10:y_int+10, z_int])
        #     anim_spot = FuncAnimation(fig_im, update_z_spot, frames=np.clip(list(range(z_int-8, z_int+8)), 0, img_fish.shape[2]), interval=200, blit=False)
        #     # plt.show()
        #     anim.save('spot_' + str(c) + '.gif', writer='imagemagick', fps=8, dpi=300)
        #     plt.close()

        # get a local block of pixels around the spot for the fit
        xyz = []
        brightness = []
        for i in range(x_int-15, x_int+15):
            for j in range(y_int-15, y_int+15):
                for k in range(z_int-8, z_int+8):
                    # make sure data is within range
                    if i >= 0 and i < img_fish.shape[0] and j >= 0 and j < img_fish.shape[1] and k >= 0 and k < img_fish.shape[2]:
                        xyz.append([i, j, k])
                        brightness.append(img_fish[i, j, k])
        
        # print('----------------')
        # print('Initial guess: x=' + str(x_int) + ', y=' + str(y_int) + ', z=' + str(z_int))

        if draw:
            fig_im, ax_im = plt.subplots()
            im = ax_im.imshow(img_fish[x_int-15:x_int+15, y_int-15:y_int+15, z_int], interpolation='bicubic', vmin=np.amin(img_fish[x_int-15:x_int+15, y_int-15:y_int+15, :]), vmax=np.amax(img_fish[x_int-15:x_int+15, y_int-15:y_int+15, :]))
       
        try:
            gauss_params, gauss_cov = optimize.curve_fit(su.gauss_3d, xyz, brightness, p0=[3., x_int, y_int, z_int, 1., 1., 0., 0., 0., 500.], bounds=([0., x_int-1, y_int-1, z_int-2, 0.5, 0.5, -np.inf, -np.inf, -np.inf, 0.], [np.inf, x_int+1, y_int+1, z_int+2, 10, 10., np.inf, np.inf, np.inf, np.inf]))
            A, x, y, z, sig_xy, sig_z, mx, my, mz, b = gauss_params
            # print(gauss_params)

            if A / baseline > snr:
                spots_filtered.append(spot)

            if draw:
                circ = plt.Circle((x-x_int+15, y-y_int+15), sig_xy, linewidth=2.5, edgecolor='r', facecolor='#00000000')
                ax_im.add_artist(circ)
                plot_circ = True
            
            spot_data.append([c] + list(spot[0:3]) + [A, A/baseline])

        except RuntimeError:
            # optimization failed
            # print('WARNING: Gaussian fit failed. Moving to next spot...')
            
            if draw:
                plot_circ = False

        if draw:
            anim = FuncAnimation(fig_im, update_z_spot, frames=np.clip(list(range(z_int-8, z_int+8)), 0, img_fish.shape[2]-1), interval=200, blit=False)
            # plt.show()
            anim.save('spot_' + str(c) + '_fit.gif', writer='imagemagick', fps=8, dpi=300)
            plt.close()

        del xyz[:]
        del brightness[:]

    return np.row_stack(spots_filtered), spot_data


# deprecated ------------------------------------------------------------------#

def find_spots(img_fish, sigma=1., t_spot=0.025, mask=None):
    """DEPRECATED: Use `find_spots_snrfilter()` instead.
    
    Find FISH spots in a 3D microscopy image.

    Find HCR-FISH spots in the FISH channel and return the positions of all 
    detected spots using `skimage.feature.blob_log`.

    Parameters:

    img_fish: np.array, 3D
        A 3D numpy array containing signal from the FISH channel.
    

    Optional:

    sigma: float (default 1.0)
        The standard deviation of the Gaussian kernel used for blurring before 
        the Laplacian operation.  Increasing this value will preferentially 
        detect larger spots.
    
    t_spot: float (default None)
        Threshold to use for spot detection, from 0 to 1. Typical values for 
        this parameter fall between 0.01 and 0.05. Lower this value to detect 
        more spots.

    mask: np.array, 3D (same dimensions as `img_fish`)
        If provided, spots detected outside the mask will not be returned.
    """

    spots = feature.blob_log(su.normalize_image(img_fish), sigma, sigma, num_sigma=1, threshold=t_spot)
    
    if mask is not None:
        spots_masked = []
        for spot in spots:
            spot_pos = tuple(spot[0:3].astype(int))
            if mask[spot_pos]:
                spots_masked.append(spot)
        spots_masked = np.row_stack(spots_masked)
        return spots_masked
    else:
        return spots