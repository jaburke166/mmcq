#============================================================================
#                       UTILITY FUNCTIONS FOR MMCQ
#============================================================================


import numpy as np
import scipy

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


def crop_ROI(img, traces, return_idx=True, norm=False, replace_outer=False):
    '''
    Crop region of interest around the choroid
    
    INPUTS:
    --------------
        img (2darray) : Grayscale OCT slice
        
        traces (np.array) : Numpy array storing pixel coordinates of the upper and lower choroid
        boundaries in xy-space.

        return_idx (bool) : If flagged, ROI image and indexes used for cropping are returned. 
        Otherwise, just the ROI image is returned.
        
        norm (bool) : If flagged, normalise ROI from 0 to 1.
        
        replace_outer (bool) : If flagged, pixels that share the same column, outside the choroid 
        in cropped result are replaced with the mean of the pixels of the choroid in that column.
                
    RETURNS:
    --------------
        roi (np.array) : Cropped OCT B-scan according to region traces.
        
        crop_idxs (2-tuple) : Pixel indexes where cropping occured to allow for reconstruction.
    '''
    # Detect if traces are in xyspace or yxspace (pixel space)
    top_chor, bot_chor = traces
    if all(np.diff(top_chor[:,0]) == 1):
        h_idx = 0
    else:
        h_idx = 1
    v_idx = abs(1-h_idx)
    top_stx, bot_stx = top_chor[0,h_idx], bot_chor[0,h_idx]
    
    # Find common indexes for boundaries to crop image with
    common_st_idx = max(top_chor[0,h_idx], bot_chor[0,h_idx])
    common_en_idx = min(top_chor[-1,h_idx], bot_chor[-1,h_idx])
    top_idx = top_chor[:,v_idx].min()
    bot_idx = bot_chor[:,v_idx].max()   
    
    # Crop image and boundaries
    ROI_img = img[top_idx:bot_idx+1, common_st_idx:common_en_idx].copy()
    shifted_topchor = top_chor[common_st_idx-top_stx:common_en_idx-top_stx]
    shifted_botchor = bot_chor[common_st_idx-bot_stx:common_en_idx-bot_stx]
    shifted_chor_bounds = np.concatenate([shifted_topchor[np.newaxis], 
                                          shifted_botchor[np.newaxis]], axis=0)  
    
    # Make sure we're in xyspace and do final shift
    shifted_chor_bounds[:,:] = shifted_chor_bounds[:,:,[h_idx, v_idx]]
    shifted_chor_bounds[:,:,1] -= top_idx 
    shifted_chor_bounds[:,:,0] -= common_st_idx
    
    # Check if pixel values above and below choroid should be replaced
    if replace_outer:
        top_bnd, bot_bnd = shifted_chor_bounds[0,:,1], shifted_chor_bounds[1,:,1]
        pixel_replace_all = np.mean(ROI_img)
        for x in range(top_bnd.shape[0]):
            pixel_replace = np.mean(ROI_img[top_bnd[x]:bot_bnd[x], x])
            ROI_img[:top_bnd[x], x] = pixel_replace
            ROI_img[bot_bnd[x]:, x] = pixel_replace
            
    # Check flag to normalise to [0, 1]
    if norm:
        ROI_img = normalise(img, (0,1), np.float64)
        
    # Output by default is cropped image and shifted boundaries
    output = [ROI_img, shifted_chor_bounds]

    # If returning indexes or not
    if return_idx:
        crop_idxs = ((top_idx, bot_idx), (common_st_idx, common_en_idx))
        output.append(crop_idxs)

    return output



def rebuild_ROI(ROI, chor_bounds, img, rect=False):
    '''
    Using ROI image, chor_bounds, rebuild original image using the original intensities outwith
    chor_bounds found in img

    INPUTS:
    -------------------
        ROI (2darray) : Region of interest image, original bounded by chor_bounds.

        chor_bounds (2-tuple) : Tuple containing yx-coordinates of RPEChor and ChorScl boundaries

        img (2darray) : Original image used to crop ROI.

        rect (bool) : If flagged, ROI is built back into image as a rectangle. Otherwise, only the
        choroid is built back in.  
    '''
    # swap axes if traces are in xy-space, not pixel space
    chor_bounds = list(chor_bounds)
        
    # Detect if traces are in xyspace or yxspace (pixel space)
    top_chor, bot_chor = chor_bounds
    if all(np.diff(top_chor[:,0]) == 1):
        top_chor = top_chor[:,[1,0]]
        bot_chor = bot_chor[:,[1,0]]
        chor_bounds = [top_chor, bot_chor]

    # Determine pixel index marking common horizontal and vertical locations of traces,
    # i.e. where the image was originally cropped to.
    top_st = chor_bounds[0][0,1]
    bot_st = chor_bounds[1][0,1]
    common_st_idx = max(chor_bounds[0][0,1], chor_bounds[1][0,1])
    common_en_idx = min(chor_bounds[0][-1,1], chor_bounds[1][-1,1])
    
    # Shift the boundaries according to these common start and end pixel indexes
    # After shifting, work out minimum and maximum choroid height and crop accordingly
    shifted_rpechor = chor_bounds[0][common_st_idx-top_st:common_en_idx-top_st].copy()
    shifted_chorscl = chor_bounds[1][common_st_idx-bot_st:common_en_idx-bot_st].copy()
    top_idx = shifted_rpechor[:,0].min()
    bot_idx = shifted_chorscl[:,0].max()

    # Copy image as output and put enhanced OCT into image
    output = img.copy()
    if not rect:
        x_roi = np.arange(common_en_idx-common_st_idx)
        top_bnd, bot_bnd = shifted_rpechor[:,0], shifted_chorscl[:,0]
        for x in x_roi:
            output[top_bnd[x]:bot_bnd[x], x+common_st_idx] = ROI[top_bnd[x]-top_idx:bot_bnd[x]-top_idx, x]
    else:
        output[top_idx:bot_idx, common_st_idx:common_en_idx] = ROI
    
    return output


def sigmoid(x, a=1, b=0.5, out=1): 
    '''
    Define sigmoid function between [0, 1] X [0, out]. The typical formula has been adapted so that sigmoid(0) ~ 0,
    sigmoid(b) = out//2 and sigmoid(1) ~ out, with default hyperparameters a=1 and b=0.
    
    Currently only a can be changed in main algorithm. In future versions, b could too. Sensible values for b
    are [0.25, 0.75] (b performs horizontal shifting).

    INPUTS:
    -----------------
        x (array) : 1D array of values in [0, rng[0]].

        a (float, default 1) : Multiplicative coefficient representing steepness/sharpness of sigmoid curve.

        b (float, default 0.5) : Additive coefficient representing shift in x-axis of sigmoid curve. This parameter
        defines the x-value where the sigmoid function is equal to half the range.
        
        out (integer, default 1) : Maximum value range.
    '''
    # Compute exponential variable
    multi = 2*np.pi**2*a
    exp_var = multi*(x-b)
    
    # Return sigmoid in [0, y_max] range.
    return out/(1+np.exp(-exp_var))


def sigmoid_ada(a0=2.0, a1=0.5, b=0.5, out=1): 
    '''
    Adapted sigmoid function between [0, 1] X [0, out]. The typical formula has been adapted so that sigmoid(0) ~ 0,
    sigmoid(b) = out//2 and sigmoid(1) ~ out, and has fixed sharpness parameters a0 and a1 for the sigmoid segments
    either side of the shift parameter b.
    
    Shift parameter b can be altered with to define the dispersion values where most enhancement will occur. Set as 0.5,
    so that mostly homogeneous patches aren't enhanced, while inhomogeneous ones are enhanced steadily. Dispersion
    calculated as the standard deviation of the patches over the standard deviation of the image.

    INPUTS:
    -----------------
        a0 (float, default 1) : Multiplicative coefficient representing steepness/sharpness of sigmoid curve before
        hitting shift parameter b.
        
        a1 (float, default 1) : Multiplicative coefficient representing steepness/sharpness of sigmoid curve after 
        shift parameter b.

        b (float, default 0.5) : Additive coefficient representing shift in x-axis of sigmoid curve. This parameter
        defines the x-value where the sigmoid function is equal to half the range.
        
        out (integer, default 1) : Maximum value range.
    '''
    # Number of points to interpolate with
    N = 500
    
    # Compute exponential variable for first segment
    x0 = np.linspace(0,b,int(b*N))
    multi0 = 2*np.pi**2*a0
    exp_var0 = multi0*(x0-b)
    sig0 = out/(1+np.exp(-exp_var0))
    
    # Compute exponential variable for second segment 
    x1 = np.linspace(b,1,N-x0.shape[0])
    multi1 = 2*np.pi**2*a1
    exp_var1 = multi1*(x1-b)
    sig1 = out/(1+np.exp(-exp_var1))
    
    # Interpolate linearly between two segmented to get function
    x_grid = np.linspace(0,1,N)
    sig_curve = scipy.interpolate.interp1d(x_grid, np.concatenate([sig0, sig1], axis=0))
    
    # Return sigmoid interpolated function to evaluate dispersion values at.
    return sig_curve


def median_cut(features, spatial_coords, n_clusters, size=None):
    '''
    Perform median cut on a patch. 

    INPUTS:
    ---------------------
        features (np.array) : Array of pixel intensities of patch. Assumed to be 
            one dimensional.
            
        spatial_coords (ndarray) : Array storing the pixel coordinates defining the choroidal region of interest. 
            This is so we can rebuild the cluster result.

        n_clusters (int) : Number of clusters.

        size (tuple) : Size of image to put cluster IDs into according to spatial_coords. If None, then
            we are quantising a patch.

    RETURNS:
    ---------------------
        mc_predict (np.array) : Median cut clustered patch.
    ''' 
    # Initialise cluster count, i.e. we begin algorithm with all pixels in the same cluster.
    # Initialise list of pixel intensities and spatial coordinates 
    cluster_count = 1
    mc_predict = np.zeros_like(features) if size is None else np.zeros(size)
    gray_list = [features]
    spatial_list = [spatial_coords]

    # while cluster count is less than the number clusters
    while cluster_count < n_clusters:

        # Loop over patch and pixel indexes
        loop_grayscale = []
        loop_spatial = []
        for i, grayscale in enumerate(gray_list):

            # extract spatial patch
            xy_coords = spatial_list[i]

            # Compute median and split pixels 
            median = np.median(grayscale)
            pixel_below_idx = np.argwhere(grayscale <= median)[:,0]
            pixel_above_idx = np.argwhere(grayscale > median)[:,0]

            # If spliting by median results in an empty array, continue to next iteration
            above_size = pixel_above_idx.size
            below_size = pixel_below_idx.size
            if above_size * below_size > 0:
                # Append loop_grayscale
                loop_grayscale.append(grayscale[pixel_below_idx])
                loop_grayscale.append(grayscale[pixel_above_idx])

                # Append loop_spatial
                loop_spatial.append(xy_coords[pixel_below_idx])
                loop_spatial.append(xy_coords[pixel_above_idx])

            elif above_size == 0:
                loop_grayscale.append(grayscale[pixel_below_idx])
                loop_spatial.append(xy_coords[pixel_below_idx])

            elif below_size == 0:
                loop_grayscale.append(grayscale[pixel_above_idx])
                loop_spatial.append(xy_coords[pixel_above_idx])

            # Update cluster_count
            cluster_count += 1

            # Check to see if cluster number has reached n_clusters
            if cluster_count == n_clusters:
                loop_grayscale.extend(gray_list[i+1:])
                loop_spatial.extend(spatial_list[i+1:])
                break

        # update gray_list and idx_list
        gray_list = loop_grayscale
        spatial_list = loop_spatial

    # Output cluster IDs. If size is None, we are quantising 1d patches during
    # enhancement. If size is specified, we are quantising a large and irregular region,
    # such as an entire choroid.
    if size is None:
        for j, pixel_idx in enumerate(spatial_list):
            mc_predict[pixel_idx] = j+1
    else:
        for j, pixel_idx in enumerate(spatial_list):
             mc_predict[pixel_idx[:,1], pixel_idx[:,0]] = j+1

    return mc_predict
