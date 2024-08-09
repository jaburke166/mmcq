import PIL.Image as Image
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
import numpy as np
from scipy.ndimage import convolve, minimum_filter
from skimage import metrics, restoration, metrics, filters, exposure, measure
import skimage.morphology as morph

#============================================================================
#                       PREPROCESSING
#============================================================================


def normalise(img, 
              minmax_val=(0,1), 
              astyp=np.float64):
    '''
    Normalise image between minmax_val.

    INPUTS:
    ----------------
        img (np.array, dtype=?) : Input image of some data type.

        minmax_val (tuple) : Tuple storing minimum and maximum value to normalise image with.

        astyp (data type) : What data type to store normalised image.
    
    RETURNS:
    ----------------
        img (np.array, dtype=astyp) : Normalised image in minmax_val.
    '''
    # Extract minimum and maximum values
    min_val, max_val = minmax_val

    # Convert to float type to perform [0, 1] normalisation
    img = img.astype(np.float64)

    # Normalise to [0, 1]
    img -= img.min()
    img /= img.max()

    # Rescale to max_val and output as specified data type
    img *= (max_val - min_val)
    img += min_val

    return img.astype(astyp)


def get_clahe(image, 
              clip_lim=2.0, 
              tile_size=(8, 8)):
    '''
    Contrast enhance an image using Contrast Limited Adaptive Histogram Equalisation.

    INPUTS:
    -----------------
        image (np.array, dtype=np.float64) : 2-d array of image to be enhanced.

        clip_lim (float) : Clip limit to truncate the amount of contrast stretching.

        tile_size (tuple) : Integer tuple which splits the image into grids to perform HE in each cell. These cells
        are then combined using bilinear interpolation.
            
    RETURNS:
    ----------------
        output (np.array, dtype=np.float64) : CLAHE image normalised to [0, 1]
    '''
    # Copy the image and convert intensity range to [0, 255]
    image = normalise(image, (0, 255), np.uint8)

    # Create CLAHE instance and apply to image
    clahe = cv2.createCLAHE(clipLimit=clip_lim,
                            tileGridSize=tile_size)
    equalized = clahe.apply(image)
    
    # Convert to float64 and normalise to [0, 1]
    output = normalise(equalized, (0.0, 1.0), np.float64)

    return output


def denoise(image, technique, kwargs):
    '''This function denoises an image using various algorithms specified by technique.
    
    INPUT:
    ---------
        image (2darray): Grayscale image to be denoised
            
        technique (str) : Type of denoising algorithm to fit.  
    ''' 
    if technique == 'nl_means':
        if "h" in list(kwargs.keys()):
            sigma = kwargs["h"]
            del kwargs["h"]
        else:
            sigma = restoration.estimate_sigma(image)*1.5
        denoised_img = restoration.denoise_nl_means(image, h=sigma, **kwargs)
        return denoised_img
        
    else:
        if image.ndim == 2:
            fltr = morph.disk(**kwargs)
        elif image.ndim == 3:
            fltr = morph.ball(**kwargs)
        else:
            print("Image data must be 2d or 3d input")
            return None
    
    
        
    if technique == 'median':
        image = filters.median(image.astype(np.float64), fltr)
        denoised_img = image / image.max()
        
    elif technique == "wavelet":
        sigma = restoration.estimate_sigma(image)*1.5
        denoised_img = restoration.denoise_wavelet(image.astype(np.float64), sigma=sigma)
 
    elif technique == "bilateral":
        sigma = restoration.estimate_sigma(image)*3
        denoised_img = restoration.denoise_bilateral(image.astype(np.float64), sigma_color=sigma)
        
    elif technique == 'minimum':
        denoised_img = minimum_filter(image, footprint=fltr)
        
    else:
        print(f"Denoising technique {technique} not implemented.")
        denoised_img=None

    return denoised_img



#============================================================================
#                       IMAGE LOADING AND PLOTTING
#============================================================================


def load_img(path, ycutoff=0, xcutoff=0):
    '''
    Helper function to load in image and crop
    '''
    img = np.array(Image.open(path))[ycutoff:, xcutoff:]/255.0
    ndim = img.ndim
    M, N = img.shape[:2]
    pad_M = (32 - M%32) % 32
    pad_N = (32 - N%32) % 32

    # Assuming third color channel is last axis
    if ndim == 2:
        return np.pad(img, ((0, pad_M), (0, pad_N)))
    else: 
        return np.pad(img, ((0, pad_M), (0, pad_N), (0,0)))


def generate_imgmask(mask, thresh=None, cmap=0):
    '''
    Given a prediction mask Returns a plottable mask
    '''
    # Threshold
    pred_mask = mask.copy()
    if thresh is not None:
        pred_mask[pred_mask < thresh] = 0
        pred_mask[pred_mask >= thresh] = 1
    max_val = pred_mask.max()
    
    # Compute plottable cmap using transparency RGBA image.
    trans = max_val*((pred_mask > 0).astype(int)[...,np.newaxis])
    if cmap is not None:
        rgbmap = np.zeros((*mask.shape,3))
        rgbmap[...,cmap] = pred_mask
    else:
        rgbmap = np.transpose(3*[pred_mask], (1,2,0))
    pred_mask_plot = np.concatenate([rgbmap,trans], axis=-1)
    
    return pred_mask_plot


def plot_img(img_data, traces=None, cmap=None, save_path=None, fname=None, sidebyside=False, rnfl=False):
    '''
    Helper function to plot the result - plot the image, traces, colourmap, etc.
    '''
    img = img_data.copy().astype(np.float64)
    img -= img.min()
    img /= img.max()
    M, N = img.shape
    
    if rnfl:
        figsize=(15,6)
    else:
        figsize=(6,6)

    if sidebyside:
        figsize = (2*figsize[0], figsize[1])
    
    if sidebyside:
        fig, (ax0, ax) = plt.subplots(1,2,figsize=figsize)
        ax0.imshow(img, cmap="gray", zorder=1, vmin=0, vmax=1)
        ax0.set_xticklabels([])
        ax0.set_yticklabels([])
        ax0.tick_params(axis='both', which='both', bottom=False,left=False, labelbottom=False)
    else:
        fig, ax = plt.subplots(1,1,figsize=figsize)
        
    ax.imshow(img, cmap="gray", zorder=1, vmin=0, vmax=1)
    fontsize=16
    if traces is not None:
        if len(traces) == 2:
            for tr in traces:
                 ax.plot(tr[:,0], tr[:,1], c="r", linestyle="--",
                    linewidth=2, zorder=3)
        else:
            ax.plot(traces[:,0], traces[:,1], c="r", linestyle="--",
                    linewidth=2, zorder=3)

    if cmap is not None:
        cmap_data = cmap.copy().astype(np.float64)
        cmap_data -= cmap_data.min()
        cmap_data /= cmap_data.max()
        ax.imshow(cmap_data, alpha=0.5, zorder=2)
    if fname is not None:
        ax.set_title(fname, fontsize=15)
            
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='both', bottom=False,left=False, labelbottom=False)
    ax.axis([0, N-1, M-1, 0])
    fig.tight_layout(pad = 0)
    if save_path is not None and fname is not None:
        ax.set_title(None)
        fig.savefig(os.path.join(save_path, f"{fname}.png"), bbox_inches="tight", pad_inches=0)



#============================================================================
#                       UTILIY FUNCTIONS FOR REGION SEGMENTATION
#============================================================================
def extract_bounds(mask):
    '''
    Given a binary mask, return the top and bottom boundaries, 
    assuming the segmentation is fully-connected.
    '''
    # Stack of indexes where mask has predicted 1
    where_ones = np.vstack(np.where(mask.T)).T
    
    # Sort along horizontal axis and extract indexes where differences are
    sort_idxs = np.argwhere(np.diff(where_ones[:,0]))
    
    # Top and bottom bounds are either at these indexes or consecutive locations.
    bot_bounds = np.concatenate([where_ones[sort_idxs].squeeze(),
                                 where_ones[-1,np.newaxis]], axis=0)
    top_bounds = np.concatenate([where_ones[0,np.newaxis],
                                 where_ones[sort_idxs+1].squeeze()], axis=0)
    
    return (top_bounds, bot_bounds)


def interp_trace(traces, align=True):
    '''
    Quick helper function to make sure every trace is evaluated 
    across every x-value that it's length covers.
    '''
    new_traces = []
    for i in range(2):
        tr = traces[i]  
        min_x, max_x = (tr[:,0].min(), tr[:,0].max())
        x_grid = np.arange(min_x, max_x)
        y_interp = np.interp(x_grid, tr[:,0], tr[:,1]).astype(int)
        interp_trace = np.concatenate([x_grid.reshape(-1,1), y_interp.reshape(-1,1)], axis=1)
        new_traces.append(interp_trace)

    # Crop traces to make sure they are aligned
    if align:
        top, bot = new_traces
        h_idx=0
        top_stx, bot_stx = top[0,h_idx], bot[0,h_idx]
        common_st_idx = max(top[0,h_idx], bot[0,h_idx])
        common_en_idx = min(top[-1,h_idx], bot[-1,h_idx])
        shifted_top = top[common_st_idx-top_stx:common_en_idx-top_stx]
        shifted_bot = bot[common_st_idx-bot_stx:common_en_idx-bot_stx]
        new_traces = (shifted_top, shifted_bot)

    return tuple(new_traces)


def smart_crop(traces, check_idx=20, ythresh=1, align=True):
    '''
    Instead of defining an offset to check for and crop in utils.crop_trace(), which
    may depend on the size of the choroid itself, this checks to make sure that adjacent
    changes in the y-values of each trace are small, defined by ythresh.
    '''
    cropped_tr = []
    for i in range(2):
        _chor = traces[i]
        ends_l = np.argwhere(np.abs(np.diff(_chor[:check_idx,1])) > ythresh)
        ends_r = np.argwhere(np.abs(np.diff(_chor[-check_idx:,1])) > ythresh)
        if ends_r.shape[0] != 0:
            _chor = _chor[:-(check_idx-ends_r.min())]
        if ends_l.shape[0] != 0:
            _chor = _chor[ends_l.max()+1:]
        cropped_tr.append(_chor)

    return interp_trace(cropped_tr, align=align)


def select_largest_mask(binmask):
    '''
    Enforce connectivity of region segmentation
    '''
    # Look at which of the region has the largest area, and set all other regions to 0
    labels_mask = measure.label(binmask)                       
    regions = measure.regionprops(labels_mask)
    regions.sort(key=lambda x: x.area, reverse=True)
    if len(regions) > 1:
        for rg in regions[1:]:
            labels_mask[rg.coords[:,0], rg.coords[:,1]] = 0
    labels_mask[labels_mask!=0] = 1

    return labels_mask
    

def get_trace(pred_mask, threshold=0.5, align=False):
    '''
    Helper function to extract traces from a prediction mask. 
    This thresholds the mask, selects the largest mask, extracts upper
    and lower bounds of the mask and crops any endpoints which aren't continuous.
    '''
    binmask = (pred_mask > threshold).astype(int)
    binmask = select_largest_mask(binmask)
    traces = extract_bounds(binmask)
    traces = smart_crop(traces, align=align)
    return traces


def crop_trace(traces, check_idx=60, offset=10):
    '''
    Crop trace left and right by searching check_idx either side to ensure
    that there is at least an offset pixel height
    '''
    # Align traces horizontally
    h_idx = 0
    top,bot = traces
    top_stx, bot_stx = top[0,h_idx], bot[0,h_idx]
    common_st_idx = max(top[0,h_idx], bot[0,h_idx])
    common_en_idx = min(top[-1,h_idx], bot[-1,h_idx])
    shifted_top = top[common_st_idx-top_stx:common_en_idx-top_stx]
    shifted_bot = bot[common_st_idx-bot_stx:common_en_idx-bot_stx]

    # Now check to make sure at least an offset pixel height difference
    height_idx = shifted_bot[:,1] - shifted_top[:,1]
    left_idx = np.where(height_idx[:check_idx] >= offset)[0]
    right_idx = np.where(height_idx[-check_idx:] >= offset)[0]-check_idx

    # Reconstruct bounds
    new_top = np.concatenate([top[left_idx], top[check_idx:-check_idx], top[right_idx]], axis=0)
    new_bot = np.concatenate([bot[left_idx], bot[check_idx:-check_idx], bot[right_idx]], axis=0)
    new_trace = np.asarray(interp_trace(np.concatenate([new_top[np.newaxis], new_bot[np.newaxis]], axis=0)))

    return new_trace


def offset_trace(shifted_bnds, delta_u=0, delta_l=0, delta_lr=None):
    '''
    For medium-sized vessels, we offset the upper and lower boundaries to extract
    the middle-region of the choroid.
    '''
    new_bnds = crop_trace(interp_trace(shifted_bnds, align=False), check_idx=30, offset=10)
    new_bnds[0,:,1] += delta_u
    new_bnds[1,:,1] += delta_l

    # Offset trace left and right
    if delta_lr is not None:
        l, r = delta_lr
        if r is not None:
            new_bnds = new_bnds[:,:r]
        if l is not None:
            new_bnds = new_bnds[:,l:]
            new_bnds[:,:,0] -= l
    
    # Work where this duplicated bound may be below the original lower boundary
    outlier_idxs = np.where(new_bnds[1,:,1] <= new_bnds[0,:,1]+5)

    # Replace these outlier coordinates with the lower boundary
    new_bnds[1,outlier_idxs,1] = new_bnds[0,outlier_idxs,1]+5
    
    return crop_trace(new_bnds)



def taper_duplicated_upper(shifted_bnds, delta):
    '''
    For segmentation of the choriocapillaris, we want to make sure that the 
    duplicated upper bound (offset by delta) does not intersect with the lower bound,
    so taper the duplicate to the lower bound is this happens.
    '''
    # Compute duplicated upper boundary, shifted downward by delta
    top, bot = shifted_bnds
    small_bnds = shifted_bnds.copy()
    small_bnds[1,:,1] = small_bnds[0,:,1]+delta

    # Work where this duplicated bound may be below the original lower boundary
    outlier_idxs = np.where(small_bnds[1,:,1] > shifted_bnds[1,:,1])

    # Replace these outlier coordinates with the lower boundary
    small_bnds[1,outlier_idxs] = shifted_bnds[1, outlier_idxs]

    return small_bnds



def rebuild_mask(traces, img_shape=None):
    '''
    Rebuild binary mask from choroid traces
    '''
    # Work out extremal coordinates of traces
    top_chor, bot_chor = traces
    common_st_idx = np.maximum(top_chor[0,0], bot_chor[0,0])
    common_en_idx = np.minimum(top_chor[-1,0], bot_chor[-1,0])
    top_idx = top_chor[:,1].min()
    bot_idx = bot_chor[:,1].max()

    if img_shape is not None:
        binmask = np.zeros(img_shape)
    else:
        binmask = np.zeros((bot_idx+100, common_en_idx+100))

    for i in range(common_st_idx, common_en_idx):
        top_i = top_chor[i-common_st_idx,1]
        bot_i = bot_chor[i-common_st_idx,1]
        binmask[top_i:bot_i,i] = 1

    return binmask
#============================================================================
#                       UTILIY FUNCTIONS FOR VESSEL SEGMENTATION
#============================================================================


def rebuild_map(vessel_bmap, size, crop_idxs):
    '''
    Given a 4d colourmap of a segmented contextual ROI, rebuild back to original image.
    '''
    # Extract information about contextual region and original image
    M, N = size
    m, n = vessel_bmap.shape
    (top_chor_idx, bot_chor_idx), (common_st_idx, common_en_idx) = crop_idxs
    
    # Rebuild ROI cmap vertically
    image_bmap = np.concatenate([np.zeros(shape=(top_chor_idx, n)),
                                 vessel_bmap, 
                                 np.zeros(shape=(M-bot_chor_idx-1, n))], axis=0)

    # Rebuild ROI cmap horizontally
    interim_M = image_bmap.shape[0]
    image_bmap = np.concatenate([np.zeros(shape=(interim_M, common_st_idx)),
                                 image_bmap,
                                 np.zeros(shape=(interim_M, N-common_en_idx))], axis=1)
    
    return image_bmap



def get_features(image, boundaries, stx=0, norm=False):
    '''
    Extract spatial and grayscale features from image

    INPUTS:
    ---------------
        Image (2darray) : Image array

        boundaries (tuple) : Tuple storing the upper and lower choroid boundaries
        
        norm (bool) : If flagged, normalise the spatial coordinates
    '''

    # Extract size of image and upper and lower choroid boundaries
    img_height, img_width = image.shape
    top_chor, bot_chor = boundaries
    N = top_chor.shape[0]
    assert N == bot_chor.shape[0], "Traces must have same length"

    # Extract grayscale intensities within upper and lower choroid boundaries and store spatial coordinates of
    # grayscale intensities 
    pixel_x = []
    pixel_y = []
    grayscale_lst = []
    for i in range(N):
        grayscale_lst.append(image[top_chor[i]:bot_chor[i], i+stx])
        pixel_y.append(np.arange(top_chor[i], bot_chor[i]))
        pixel_x.append(np.asarray(pixel_y[i].shape[0]*[i+stx]))

    # Concatenate list of coordinates spaially and grayscale
    pixel_x = np.concatenate([pixel_x[i] for i in range(0, N)], 0)
    pixel_y = np.concatenate([pixel_y[i] for i in range(0, N)], 0) 
    grayscale = np.concatenate([grayscale_lst[i] for i in range(0, N)], 0)
    spatial_coord = np.concatenate([pixel_x[:, np.newaxis], pixel_y[:, np.newaxis]], axis=1).astype(np.int64)
    if norm:
        spatial_coord = spatial_coord.astype(np.float64)
        spatial_coord[:,0] /= img_width
        spatial_coord[:,1] /= img_height

    return grayscale, spatial_coord



def create_cluster_masks(cluster_image):
    '''
    Show cluster results and overlay onto original/enhanced image
    '''
    # Compute number of clusters and locate pixel coordinates per cluster
    N_cluster = np.unique(cluster_image).shape[0]-1
    background = np.argwhere(cluster_image == 0)
    cluster_idx = [np.argwhere(cluster_image == i) for i in range(1, N_cluster+1)]

    cl_imgs = [np.zeros_like(cluster_image) for i in range(N_cluster)]
    for i, cl_idx in enumerate(cluster_idx):
        cl_imgs[i][cl_idx[:,0], cl_idx[:,1]] = 1

    return cluster_idx, cl_imgs



def morphological_filter(binmap, footprint, type="open"):
    '''
    Storing morphological filter variants based on opening and closing
    Equations come from https://www.ni.com/docs/en-US/bundle/ni-vision-concepts-help/page/grayscale_morphology.html
    '''
    
    # Proper opening
    c = morph.closing(binmap, footprint)
    o_c = morph.opening(c, footprint)
    c_o_c = morph.closing(o_c, footprint)
    OPEN = np.minimum(binmap, c_o_c)

    # Proper closing
    o = morph.opening(binmap, footprint)
    c_o = morph.closing(c, footprint)
    o_c_o = morph.opening(o_c, footprint)
    CLOSE = np.maximum(binmap, c_o_c)

    # Return depending on specified type
    if type=="open":
        return OPEN
    elif type=="close":
        return CLOSE
    elif type=="automedian":
        return np.maximum(o_c_o, OPEN)




##############################################################################################################
                        # SHADOW COMPENSATION OF CHOROIDAL SPACE
##############################################################################################################


def shadow_compensate(img, gamma=1, win_size=75, plot=False):
    """Using moving averages, compensate for vessel shadowing by
    scaling A-scan pixel intensities to average out drop in signal caused by shadowing.
    Gamma is used here as an implicit enhancer too."""
    # If RGB, select first channel on the assumption it is an OCT B-scan 
    # and is therefore grayscale. Channels in last dimension by assumption.
    if img.ndim > 2:
        img = img[...,0]
    
    # Remove any black columns either side of image
    comp_idx_l = img[:,:img.shape[1]//2].mean(axis=0) != 0
    comp_idx_r = img[:,img.shape[1]//2:].mean(axis=0) != 0
    img_crop = img[:,np.concatenate([comp_idx_l, comp_idx_r], axis=0)]

    # Energy of each pixel of the A-line, where gamma > 1 for implicit image enhancement
    # by darkening the image
    E_ij = exposure.adjust_gamma(img_crop, gamma=gamma)

    # Total energy of each A-scan is the sum across their rows
    E_i = E_ij.sum(axis=0)

    # Centred, moving average according to win_size, pad edges of average with original signal
    E_i_smooth = pd.Series(E_i).rolling(win_size, center=True).mean().values
    E_i_smooth[:win_size//2], E_i_smooth[-win_size//2:] = E_i[:win_size//2], E_i[-win_size//2:]

    # Compensation is linear scale made to individual energy levels to match total energy per A-scan
    # with its moving average value
    E_comp = (E_i_smooth / E_i)

    # If plotting energy levels
    if plot:
        fig, (ax, ax1) = plt.subplots(1,2,figsize=(14,7))
        ax.set_xlabel("A-scan (column) index", fontsize=14)
        ax.set_ylabel(f"Energy level (I$^\gamma$)", fontsize=14)
        ax.plot(np.arange(E_i.shape[0]), E_i, c="b", linewidth=2)
        ax.plot(np.arange(E_i_smooth.shape[0]), E_i_smooth, c="r", linewidth=2)
        ax1.set_xlabel("A-scan (column) index", fontsize=14)
        ax1.set_ylabel(f"Correction factor (I$^\gamma$ / MA(I$^\gamma$)", fontsize=14)
        ax1.plot(np.arange(E_comp.shape[0]), E_comp, c="r", linewidth=2)
        ax.set_title(f"Energy per A-scan ($\gamma$ = {gamma})", fontsize=18)
        ax1.set_title(f"Correction per A-scan ($\gamma$ = {gamma})", fontsize=18)

    # Reshape to apply element-wise to original image and darken
    E_comp_arr = E_comp.reshape(1,-1).repeat(img_crop.shape[0], axis=0)
    output = E_ij*E_comp_arr
    #output = (img_crop*E_comp_arr)**gamma

    # Put back any black columns either side of image
    if (~comp_idx_l).sum() > 0:
        output = np.pad(output, ((0,0),((~comp_idx_l).sum(),0)))
    if (~comp_idx_r).sum() > 0:
        output = np.pad(output, ((0,0),(0,(~comp_idx_r).sum())))

    # Plot original and compensated versions
    if plot:
        fig, (ax, ax1) = plt.subplots(1,2,figsize=(18,7))
        ax.imshow(img, cmap="gray")
        ax1.imshow(output, cmap="gray")
        ax.set_axis_off()
        ax1.set_axis_off()
        fig.tight_layout()

    return output

