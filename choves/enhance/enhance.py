import numpy as np
import skimage
import functools
import operator
import scipy
from .utils import crop_ROI, rebuild_ROI, sigmoid, sigmoid_ada, median_cut, normalise

#============================================================================
#                               MMCQ
#============================================================================


class Multiscale_MCQ(object):
    '''
    This module performs multi-scale median cut quantisation for multi-scale
    feature enhancement. This particular version is applied to choroid vasculature in OCT data
    '''

    def __init__(self, 
                 image, 
                 oct_boundaries=None,
                 n_scales=None
                ):
        '''
        Initialisation of multiscale enhancement and quantisation using median cut clustering
        algorithm.
        
        In order to remove temptation to alter parameter setting, these are fixed. Those original
        parameters are describes below but are tagged with #.

        INPUTS:
        ------------------------
            img (2darray) : Array storing grayscale pixel intensities of image, in [0, 1].

            oct_boundaries (tuple) : 2-tuple storing the coordinates outlining the region of interest to 
                quantise. If None, the entire image is quantised at multiple scales. First tuple is upper 
                horizontal boundary, second is lower horizontal boundary. 
                
            n_scales : Number of scales to perform quantisation at. This can be specified in multiple ways.
                - (int) : List of n patch sizes are generated based on the ROI's height (since choroid is always
                    wider than it is longer, so features are better represented according to axial resolution).
                - (default) : Patch size default list based on step size self.step.
                - (custom) : Custom list of tuple patch sizes, [(h1, w1), ..., (hn, wn)]

            #strength (float) : Strength of enhancement in [0,1] by varying the sharpness of the sigmoid curve.
            #    Values closer to 1 performs strong enhancement for non-homogeneous
            #    patches and little enhancement for homogeneous patches. Default value is strength=1     
            
            #shift (float) : Shift parameter for sigmoid function, defines coefficient representing shift in x-axis 
            #    of sigmoid curve. This parameter defines the x-value where the sigmoid function is equal to half the range. 
            #    This can be interpreted as the turning point between low enhancement and high enhancement. Default value
            #    is shift=0.5,

           #n_subints (int) : Number of subintervals to bin individual patches into to deduce number of
           #     clusters for each patch using median cut. Default it 8 (8 possible quantisations 
           #     per patch). This value changes granularity of patch quantisation. Default is 10.
                
           # nbin (int, default 2**8) : Output number of grayscale levels (output bit depth assumed as 8).
            
           # NR_OF_GRAY (int, default 2**14) : Enhancement grayscale levels (enhancement bit depth assumed as 14). 
           #     The higher the value here implies the more granular the enhancement.
        '''
        # Global parameter setting of image, scales and subints.
        # automated step size.
        self.image = image
        self.n_scales = n_scales
        self.n_subints = 10
        self.step = 10
        self.nbins = 256 
        self.NR_OF_GRAY = 2**14 #if 256 <= NR_OF_GRAY <= 2**16 else 2**16
        
        # Define sigmnoid curve used for weighting patch enhancements - sharpness of sigmoid curve either side of
        # shift parameter fixed for now, rendering self.strength useless. This adaptive sigmoid curve now place very
        # little enhancement on mostly homogeneous patches, and strong enhancement on most inhomogeneous patches. 
        self.sigmoid_curve = sigmoid_ada(a0=2.0, a1=0.5, b=0.5)
        
        # If specified, crop OCT B-scan using region traces to enhance only choroidal contextual ROI
        if oct_boundaries is not None:
            self.provided_oct = True
            rpechor, chorscl = oct_boundaries     
            output = crop_ROI(image, oct_boundaries, return_idx=True)
            self.img, self.oct_boundaries, crop_idxs = output
            
        # If None, then enhancing entire input image
        else:
            self.provided_oct = False
            M, N = image.shape
            self.rpechor = np.concatenate([np.zeros((N,1)),np.arange(N)[:,np.newaxis]], axis=1)
            self.chorscl = np.concatenate([(M-1)*np.ones((N,1)), np.arange(N)[:,np.newaxis]], axis=1)
            self.oct_boundaries = np.concatenate([self.rpechor.reshape(1,-1,2), self.chorscl.reshape(1,-1,2)], axis=0) 
            self.img = image
        
        # ndim and shape
        self.ndim = self.img.ndim
        self.img_shape = self.img.shape
        
        # Convert image pixel intensity from float range [0, 1] to integer range [0, self.NR_OF_GRAY-1].
        # This is only necessary if nbins != 256 (the input bit depth of the image), While nbins is fixed at 256
        # we leave this rescaling of bit depth (and the code in pad_lut_image() to scale back to nbins) for completeness.
        image_int = skimage.img_as_uint(self.img)
        self.image_int = skimage.exposure.rescale_intensity(image_int, out_range=(0, self.NR_OF_GRAY-1)).astype(np.uint16)
        
        # Approximate intensity dispersion of image using standard deviation of image
        image_nbins = skimage.exposure.rescale_intensity(image_int, out_range=(0, self.nbins - 1)).astype(np.uint16)
        self.image_max = image_nbins.max()
        self.img_std = np.std(image_nbins)
        



    def pad_lut_image(self, patch_size):
        '''
        Pad image according to patch size and construct LUT between nbins and NR_OF_GRAY graylevels.
        
        This implementation of image padding and grayscale transformation follows the Skimage 
        python implementation for CLAHE: 
        https://github.com/scikit-image/scikit-image/blob/main/skimage/exposure/_adapthist.py#L26-L95
        
        INPUTS:
        ------------------------            
            patch_size (2-tuple) : Tuple storing the height and width of a patch.

        RETURNS:
        ------------------------
            image_ (np.array) : Padded image with bit depth dictated by nbins.
            
            (pad_st_idx, pad_en_idx) : Pixel indexes at the start and end of each axis at 
                which padding ends. 
        '''
        # Determine padding pixels in each dimension such that image dimension shapes are a multiple of 
        # the patch size with starting dim padding allowing patches to center on contextual region 
        # on the boundaries rather than padded region.
        pad_st_idx = [k // 2 for k in patch_size]
        pad_en_idx = [(k - s % k) % k + int(np.ceil(k / 2.)) for k, s in zip(patch_size, self.img_shape)]
        image_int_padded = np.pad(self.image_int, [[p_st, p_en] for p_st, p_en in zip(pad_st_idx, pad_en_idx)], 
                                  mode='reflect')

        # Determine gray value bin size and look up table to convert to and from bit depth of
        # NR_OF_GRAY and nbins. Apply LUT to obtain preprocessed image_ in [0, nbins-1]
        # This code is only necessary if nbins is different from the bin depth of the original image, 
        # i.e. if nbins != 256.
        bin_size = self.NR_OF_GRAY // self.nbins
        lut = np.arange(self.NR_OF_GRAY)
        lut //= bin_size
        image_ = lut[image_int_padded]
                
        return image_, (pad_st_idx, pad_en_idx)




    def extract_patches(self, image, patch_size, padded=False):
        '''
        Vectorised function to slice patches out of padded image efficiently. It works out how many 
        patches vertically and horizontally there needs to be and slices and reshapes accordingly.
        
        INPUTS:
        ------------------------
            image (ndarray) : Image to be sliced.
            
            patch_size (2-tuple) : Tuple of vertical (m) and horizontal (n) pixel length of patches.
            
            padded (bool, default False) : If flagged, all patches are extracted, including padded 
            areas. Otherwise only contextual regions (image content) are extracted.
            
        RETURNS:
        ------------------------
            all_patches (np.array) : Array of shape (N_v * N_h, m * n) containing patches extracted from
            image.
            
            patch_ordered_shape (4-tuple) : Tuple storing the original shape of the four-dimensional
            patch array, i.e. (N_v, N_h, m, n).
        
        '''
        # This defines number of patches vertical and horizontal to cover region of image. patch_slices 
        # defines the slice object which will slice image in both dimensions.
        if not padded:
            nvh_patches = [int(s / k) - 1 for s, k in zip(image.shape, patch_size)]
            patch_slices = [slice(k // 2, k // 2 + n * k) for k, n in zip(patch_size, nvh_patches)]
        else:
            nvh_patches = [int(s / k) for s, k in zip(image.shape, patch_size)]
            patch_slices = [slice(0, n*k) for k, n in zip(patch_size, nvh_patches)]
            
        # Collect nvh_patches and patch_size together, this is not ordered according to 
        # (image_dim, patch_dim) so patch_axis_order records this
        patch_unordered_shape = np.array([nvh_patches, patch_size]).T.flatten()
        patch_axis_order = np.array([np.arange(0, self.ndim * 2, 2), 
                                     np.arange(1, self.ndim * 2, 2)]).flatten()
        
        # Slices image into patches and reshape to (N_v, m, N_h, n)
        all_patches = image[tuple(patch_slices)].reshape(patch_unordered_shape)

        # Transpose axes so that shape corresponds to (N_v, N_h, m, n), store shape and finally
        # reshape to ((N_v * N_h, m * n))
        all_patches = np.transpose(all_patches, axes=patch_axis_order) 
        patch_ordered_shape = all_patches.shape 
        all_patches = all_patches.reshape((np.product(nvh_patches), -1)) 
        
        return all_patches, patch_ordered_shape    
    

    def estimate_cluster_K(self, patch):
        '''
        New, simpler version to estimate number of clusters based on patch dispersion via the standard deviation
        of the patch. Look at the spread across grayscale values 1 standard deviation either side of mean. 
        
        Number of clusters to quantise image is the number of intervals of stepsize 1/N_subints the dispersion of
        the patch's grayscale intersect. N_subints here is used to divide the [0, 1] grayscale line into equal segments
        and detect the number of clusters via the patch's dispersion.
        
        INPUTS:
         ---------------------
             patch (np.array) : Grayscale patch extracted from image with \sqrt{self.nbins}
                 bit depth, reshaped as a 1-d array.
            
         RETURNS
         ---------------------  
             n_clusters (int) : Number of clusters to quantise patch into. 
        '''
        return np.clip(np.round(2*np.std(patch/self.image_max)*self.n_subints), 2, self.n_subints)
    

    def replace_with_median(self, patch, cluster_patch):
        '''
        Using the clustered patch, replace the pixel intensities in the original patch with 
        the median of the intensities of each cluster quantising each patch.
        
        INPUTS:
        ---------------------
            patch (np.array) : 1-D array of pixel intensities in [0, nbins-1].
            
            cluster_patch (np.array) : 1-D array storing cluster indexes for patch from 
                median cut application.
            
        RETURNS:
        ---------------------
            median_patch (np.array) : 1-D array with pixel intensities belonging to the same
                cluster replaced by their median.
        '''
        # Extract cluster indexes and initialise patch to populate with median cluster values
        cluster_ids = np.unique(cluster_patch)
        median_patch = np.zeros_like(patch)
        
        # Loop over cluster IDs for patch and fill pixels corresponding to each 
        # cluster with median of the cluster it belongs to
        for cl_id in cluster_ids:
            pixel_idx = np.argwhere(cluster_patch == cl_id)
            median_patch[pixel_idx] = np.median(patch[pixel_idx])
            
        return median_patch



    def map_histogram(self, median_patches, patch_dispersions):
        '''
        This stretches each patch's histogram, actually performing the contrast enhancement. 
        Each row in median_patches represents the histogram for a patch, quantised previously 
        using median cut, shape=(N_V, N_H, nbins) where nbins defines number of graylevels. 
        The steps for this function are as follow:

        1) A cumulative histogram (CH) is calculated for each patch's histogram (PH). The CH is 
        normalised to [0,1] using n_pixels and linearly mapped/stretched to [0, NR_OF_GRAY-1]. 
        Note: this actually performs the patch enhancement.
        
        2) We also generate a cumulative histogram of the original median patch without histogram
        stretching. This is used as a way to offset the enhancement from (1).
        
        3) Weights are computed using the patch dispersions. Dispersions (proxy measure for patch 
        homogeneity) are fed into sigmoid function to determine how much enhancement each patch 
        will recieve depending on its homogeneity.
        
        4) The enhancement graylevels are combined with the original graylevels according to the 
        weighting in (4).
        
        The weighting computed in (3) will make sure to keep the enhancement for patch's whose 
        darker features we want to enhance from lighter feaures but reduce this enhancement for
        more homogeneous patch's.   

        INPUTS
        ---------------------
            median_patches (np.array) : 2-D array of median cut quantised patches of shape (N_patches, N_pixels).
            
            dispersion (np.array) : 1-D array of floats in [0, 1] estimating the homogeneity of a patch. Values
                closer to 0 correspond to homogeneous patches (lack of dispersion compared to the original image).
                
        RETURNS:
        ---------------------
            patch_enhancements (np.array) : 2-D array of enhanced cumulative histograms of the median patches
                of shape (N_patches, self.nbins).
        '''
        # Extract minimum and maximum values for graylevel mapping. This is used to construct lookup table (mapping)
        # for each patch's histogram.
        n_pixels = median_patches.shape[1]
        max_val = self.NR_OF_GRAY - 1
        
        # 1) Create cumulative distribution using the bincounts of each median patch. Stretch cumulative 
        # histogram from [0, nbins-1] to [0, NR_OF_GRAY-1]. Division by n_pixels ensures median_bincount[i].sum() = 1.0
        median_bincount = np.apply_along_axis(np.bincount, -1, median_patches, minlength=self.nbins)
        enhance =  np.cumsum(median_bincount, axis=-1).astype(float)
        enhance *= max_val / n_pixels
        
        # 2) Reconstruct cumulative histogram of median cut quantised patch, rescaled to [0, NR_OF_GRAY-1].
        # This is so if patch is relatively homogeneous, then little enhancement is done and the patch is
        # left relatively unchanged in (4)
        patch_argwheres = np.argwhere(median_bincount)
        rescale = np.zeros_like(enhance)
        for i in np.unique(patch_argwheres[:,0]):
            glevels = np.unique([0] + list(patch_argwheres[patch_argwheres[:,0]==i, 1]) + [self.nbins]).tolist()
            naggre_cumsum = [[glvl]*(glevels[j+1]-glvl) for j,glvl in enumerate(glevels[:-1])]
            rescale[i] = np.array(functools.reduce(operator.iconcat, naggre_cumsum, []))
        rescale *= max_val / (self.nbins - 1)
        
        # 3) Weights are chosen such that those patches with low dispersion (homogeneous patches) have little to no
        # enhancement and vice versa for those patches with high dispersion. Level of enhancement dictated by sigmoid.
        weights = np.repeat(self.sigmoid_curve(patch_dispersions)[:,np.newaxis], self.nbins, axis=-1)

        # 4) Generate cumulative distribution of each patch's histogram and force range to be in [0, self.NR_OF_GRAY-1]
        # ensuring to clip is any values reach higher than max_val
        result = weights * enhance + (1-weights) * rescale
        patch_enhancements = np.clip(result, a_min=None, a_max=max_val).astype(int)

        return patch_enhancements



    def fast_multilinear_interp(self, cumu_hists, patches, n_vh_patches, patch_size):
        '''
        In order to blend quantised patches, vectorised multilinear interpolation is performed using the 
        patch cumulative histograms.

        chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://rjwagner49.com/Mathematics/Interpolation.pdf
        
        INPUTS:
        ---------------------
            cumu_hists (3darray) : Array storing the each patch's cumulative histogram LUT, range of values in 
                [0, NR_OF_GRAY-1]. Size is (n_v_patches, n_h_patches, nbins)
            
            patches (2darray) : Array storing all patches of padded image. 
            Shape is (n_v_patches*n_h_patches, n_pixels)
                        
            n_vh_patches (2-tuple) : Number of patches vertically and horizontally which contains padded image 
                (padded to allow for image dimensions to be a multiple of patch_size)
            
            patch_size (2-tuple) : Tuple storing the size of patch in vertical and horizontal dimensions.
            
        RETURNS:
        ---------------------
            result (np.array) : Multilinear interpolated cumulative histograms blended together.
        '''
        # Pad cumulative histogram LUT's in image dimensions duplicating leading edges, but leave last axis (histogram values)
        # alone, i.e. map_array[0,0] = map_array[1,0] = map_array[0,1] = map_array[0,1] = cumu_hists[0,0]. 
        map_array = np.pad(cumu_hists, [[1, 1] for _ in range(self.ndim)] + [[0, 0]], mode='edge')
        
        # calculate interpolation coefficients, both direct and inverse. These are normalized according to patch_size
        # and represent multiplicative factors to help with smoothing neighbouring contextual patches together. Here,
        # patches which are closer together have a higher coefficient than those farther away.
        coeffs = np.meshgrid(*tuple([np.arange(k)/k for k in patch_size]))[::-1]
        coeffs = [np.transpose(c).flatten() for c in coeffs]
        inv_coeffs = [1 - c for c in coeffs]
        
        # Initialise result as shape (n_patches, n_pixels) to accumulate contributions from interpolation
        n_patches = np.product(n_vh_patches)
        n_pixels = np.product(patch_size)
        result = np.zeros((n_patches, n_pixels), dtype=np.float32)
        
        # Looping over each image dimension's edge indices -- i.e. for 2D we have (0,0), (1,0), (0,1), (1,1) -- lets 
        # us slice this map_array so that each iteration increments the result array using a contribution from a pixel's
        # neighbouring contextual patch - effectively performing one part of the interpolation.
        for edge in np.ndindex(*([2] * self.ndim)): # This is just (2,2) for 2d input
            
            # Slice cumulative histogram patch's according which edge we're interpolating with, This defines one of the four
            # *possible* contributions each pixel in the result array will have during multilinear interpolation. For example
            # edge=(0,0) => edge_maps = map_array[:-1, :-1] while edge=(1,0) => edge_maps = map_array[1:, :-1]. This has the 
            # effect of locating adjacent cumulative histograms in each iteration.
            edge_maps = map_array[tuple([slice(e, e + n) for e, n in zip(edge, n_vh_patches)])]
            edge_maps = edge_maps.reshape((n_patches, -1)) # shape=(n_patches, nbins)

            # Each row in patches is a patch of bit-depth nbins, and in edge_maps are the patch's corresponding LUT of new intensities 
            # from enhancing in map_histogram(), which are in bit-depth [0, NR_OF_GRAY-1] and of length nbins. So, we select from the LUT 
            # the values in edge_maps which correspond to the intensities in the patch (their location in the LUT, to be precise).
            # Therefore, each patch's new value is an enhanced intensity from edge_maps LUT (unique per patch). This is done column-wise for 
            # every row, i.e. patch-pixel-wise for every patch.
            edge_mapped = np.take_along_axis(edge_maps, patches, axis=-1) # shape (n_patches, n_pixels)
            
            # Select the interpolation coefficients so that edge_mapped * edge_coeffs performs one part of the bi-linear
            # interpolation.
            edge_coeffs = np.product([[inv_coeffs, coeffs][e][d] for d, e in enumerate(edge[::-1])], axis=0)
            
            # Increment result by applying interpolation coefficients to each patch's mapped graylevels 
            # (still in [0, NR_OF_GRAY-1]). This accumulation of different edge interpolations builds the final, blended
            # result, preventing border artifacts
            result += (edge_mapped * edge_coeffs).astype(result.dtype)
            
        return result




    def spatial_quant(self, patch_size):
        '''
        Performs spatial quantification on an image at a particular scale dictated by patch_size. The 
        boundaries parameter us used to set all pixel intensities outwith these boundaries to their 
        original pixel intensity value pre-quantisation.
        
        INPUTS:
        ---------------------
            patch_size (2-tuple) : Tuple of integers storing patch size to determine scale of 
                quantisation
            
        RETURNS:
        ---------------------
            output (np.array) : Enhanced choroid of the scale dictated by patch_size
        '''

        # Work out padding dimensions so that resolution is a multiple of the patch size in both dimensions.
        # Pad image and map pixels from NR_OF_GRAY bit depth to nbins bit depth (Only relevant if nbins is
        # different to what the input bit depth is, i.e. nbins != 256 in most cases).
        m, n = patch_size
        image, (pad_st_dim, pad_en_dim) = self.pad_lut_image(patch_size)

        # Efficiently slice patches from padded image into shape (N_patches_V, N_patches_H, m, n), excluding patches
        # lying along the top- and left-most edges.
        all_patches, patch_ordered_shape = self.extract_patches(image, patch_size, padded=False)

        # Estimate number of clusters for each patch based on each patch's histogram
        patch_cluster_K = np.apply_along_axis(self.estimate_cluster_K, -1, all_patches)

        # Perform median cut on each patch
        spatial_coords = np.arange(np.product(patch_size))
        clustered_patches = np.array([median_cut(patch, spatial_coords, K) for (patch, K) in zip(all_patches, patch_cluster_K)])

        # For each clustered patch, set pixel intensities of pixels in each cluster as median value of pixels
        # in said cluster
        median_patches = np.array([self.replace_with_median(patch, patch_clust)\
                                          for patch, patch_clust in zip(all_patches, clustered_patches)])
                
        # Measure of dispersion of each patch is the standard deviation of a patch divided by the total 
        # standard deviation of the whole image
        patch_dispersion = np.clip(np.apply_along_axis(np.std, -1, all_patches) / self.img_std, 0, 1)
        
        # Calculate the graylevel mapping (lookup table). It does so by forming cumulative distribution of each 
        # patch's histogram. cumu_hists is therefore just the cumulative distribution of the bin counts for every 
        # median patch, converted to bit depth NR_OF_GRAY and stretched dependent on its dispersion from above.
        # Reshape to (N_V, N_H, nbins)
        cumu_hists = self.map_histogram(median_patches, patch_dispersion).reshape(*patch_ordered_shape[:self.ndim]+(-1,))

        # Rearrange padded image into blocks for vectorized processing to blend patches together using multilinear
        # interpolation.
        padded_patches, padded_ordered_shape = self.extract_patches(image, patch_size, padded=True)

        # Convert cumulative distributions to grayscale levels and blend median patches for final, padded result
        result = self.fast_multilinear_interp(cumu_hists, padded_patches, 
                                              padded_ordered_shape[:self.ndim], patch_size)
        
        # Convert result back to original data type and reconstruct back to image shape efficiently 
        result = result.reshape(padded_ordered_shape)
        patch_axis_order = np.array([np.arange(0, self.ndim), np.arange(self.ndim, self.ndim*2)]).T.flatten()
        result = np.transpose(result, axes=patch_axis_order).reshape(image.shape)
        unpad_slices = tuple([slice(idx_s, img_s - idx_f)\
                            for idx_s, idx_f, img_s in zip(pad_st_dim, pad_en_dim, image.shape)])
        result = normalise(result[unpad_slices])
        
        # Rebuild original image with enhanced OCT 
        if self.provided_oct:
            result = rebuild_ROI(result, self.oct_boundaries, self.image, rect=False)
        
        return result        


    def __call__(self):
        '''
        Wrapper function to perform the spatial quantisation over multiple scales and combine all 
        results for a grayscale image.
        ''' 
        # Extract size of image and upper and lower choroid boundaries
        M, N = self.img_shape

        # Create patch sizes if n_scales is integer type
        if isinstance(self.n_scales, int):
            # Choose smaller square patches, i.e. M since choroid is always longer horizontally
            all_patch_sizes = np.array([(M//i, M//i) for i in range(self.n_scales, 1, -1)])
    
        # If input list of scales
        elif isinstance(self.n_scales, list):
            all_patch_sizes = self.n_scales
            
        # Default scales - choosing 3 distinct scales that are square and multiples of eachother, providing three
        # levels of enhancement which we can combine into 3 separate channels of enhancement
        elif self.n_scales is None:
            all_patch_sizes = [[(i*self.step, i*self.step), (M, M)][np.argmin([i*self.step, M])] for i in range(1,4)]

        # Initialise empty list of quantisation results and loop over patch sizes
        scale_quant = []
        for size in all_patch_sizes:
            scale_quant.append(self.spatial_quant(size))

        # Stack results together and combine through taking minimum operation at every pixel
        stacked = np.concatenate([img[np.newaxis] for img in scale_quant], axis=0)
        stacked_min = np.amin(stacked, axis=0)
         
        return stacked, stacked_min