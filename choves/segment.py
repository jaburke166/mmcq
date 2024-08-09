import skimage.exposure as exposure
import skimage.morphology as morph
import skimage.measure as meas
import numpy as np
import logging
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torch
from skimage.filters import threshold_niblack

################################################################################################
#                       VESSEL SEGMENTATION OUTER FUNCTIONS
################################################################################################

from .enhance import enhance
from .enhance.utils import crop_ROI, rebuild_ROI, median_cut
from .metrics.choroid_utils import detect_orthogonal_chorscl
from .utils import (normalise, get_clahe, denoise, extract_bounds,
                    interp_trace, crop_trace, offset_trace, taper_duplicated_upper, shadow_compensate,smart_crop,
                    get_features, rebuild_map, create_cluster_masks, morphological_filter, rebuild_mask)

def preprocess_vesselseg(img, mmcq=True, pixel_thickness=None):
    '''
    Helper function to preprocess OCT B-scan images. 
    
    We start with speckle denoising and smoothing using Non-local means.

    We then apply a Gamma brightness adjustment to darken the image to further reduce any noise.

    We then use CLAHE to adjust the contrast between luminal and stromal tissue.

    Finally, we apply MMCQ to quantise and enhance the choroidal region. This is smoothed using a median filter.
    
    We are now ready for clustering-based segmentation.
    '''
    # Fix preprocessing keyword arguments
    nl_kwargs = {'patch_size':5, 'patch_distance':10, 'h':0.015,'fast_mode':True}
    clahe_kwargs = dict(clip_lim = 2, tile_size = (8, 8))
    if pixel_thickness is None:
        mmcq_kwargs = dict(scales=[(10,10), (25,25), (40,40)])
    else:
        T = pixel_thickness
        scales = [(max(5,(T//i)), max(5,(T//i))) for i in [10, 5, 2]]
        mmcq_kwargs = dict(scales=scales)
    # print(mmcq_kwargs)
    # print(T)
    
    # Store all resultant images
    output = []

    # Compensate for any potential shadows in the OCT B-scan
    img = shadow_compensate(img)
    
    # Smoothing using NL-means
    img = denoise(img, technique='nl_means', kwargs=nl_kwargs)
    output.append(img)
    
    # Adjust contrast using CLAHE and smooth
    img = get_clahe(img, **clahe_kwargs)
    img = denoise(img, "median", dict(radius=2))
    output.append(img)

    # Darken CLAHE image
    new_mean = 0.35
    gamma = np.log(new_mean) / np.log(img.mean())
    img = exposure.adjust_gamma(img, gamma=gamma)
    output.append(img)

    # Enhance and quantise choroid and smooth
    if mmcq:
        stacked, img = enhance.Multiscale_MCQ(img, **mmcq_kwargs)()
        output.append(stacked)
        output.append(denoise(img, "median", dict(radius=2)))
        
    return output


def compute_vessel_segmentation(enhanced_img, chor_bounds, cluster_type, K, N_keep, stx=0):
    '''
    Wrapper function to tie segmentation and localisation together.
    
    `vessel_binmap` stores the binary segmentation map of vessels for each slice of the OCT volume. 
    `vessel_count` will increment with each slice the number of vessels accumulated, regardless of size. 
    `vessel_masks` will store a list of arrays with each array storing all individual binary masks of each 
        vessel for every slice.
    `vessel_pixels` will store a list of arrays with each array storing individual pixel indexes for each 
        individual vessel for every slice.
    
    INPUTS:
    --------------------
        enhanced_img (np.array) : OCT B-scan, already enhanced and cropped to only include contextual region.
        
        chor_bounds (2-tuple) : y-values of the upper and lower choroid boundary defining contextual region.
        
        cluster_type (str) : Whether to cluster vessel pixels using Kmeans or Median Cut.
        
        K (int) : Total number of clusters to group pixels.
        
        N_keep (int) : Number of clusters to keep as vasculature vs. extravasculature.
    '''
    # Extract features and image shape
    img_shape = enhanced_img.shape
    grayscale, spatial_coords = get_features(enhanced_img, chor_bounds, stx)

    # Pixel clustering using either Median Cut of K-means
    cluster_result = median_cut(grayscale, spatial_coords, K, img_shape)
    cluster_idxs, cl_masks = create_cluster_masks(cluster_result)

    # Build binary map of vasculature
    vessel_binmap = np.zeros_like(enhanced_img)
    for i in range(N_keep):
        vessel_binmap[cluster_idxs[i][:,0], cluster_idxs[i][:,1]] = 1

    return vessel_binmap



def segment_vessels(img, bnds, seg_params=(20, 11), 
                    scaley=3.87, return_preprocessed=False):
    '''
    Helper function storing core functionality for vessel segmentation
    '''
    # Choriocapillaris is ~10um thick (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2913695/)
    # denominator is the vertical pixel-to-micron scaling, on Heidelberg this is 3.87. 
    cc_delta = int(np.ceil(10 / scaley)) 
    all_vbinmaps = []
            
    # Organise trace such that it is evaluated across all valid x-coordinates
    # Crop such that it can cope with choriocapillaris and small-scale vessel
    # segmentation (10 pixel region below RPE-Choroid boundary ~ 38 microns deep)
    small_delta = 10
    traces = smart_crop(interp_trace(bnds), ythresh=1)
    roi_img, shifted_bnds, crop_idx = crop_ROI(img, traces)

    # Work out distance to segment upper-region of choroid based on average choroid thickness
    rpechor_pts = shifted_bnds[0,15:-15:2]
    chorscl_pts, rpechor_pts, _, _ = detect_orthogonal_chorscl(rpechor_pts, shifted_bnds, offset=15)
    boundary_pts = np.concatenate([rpechor_pts[np.newaxis], chorscl_pts[np.newaxis]], axis=0)
    mean_pixel_height = int(np.mean(np.abs(np.diff(boundary_pts[...,1], axis=0))))
    
    # Preprocess using 
    # - non-local means (smooth), 
    # - gamma contrast+CLAHE (brightness+enhancement)
    # - MMCQ (multi-scale enhancement + quantisation)
    preprocessed = preprocess_vesselseg(roi_img, pixel_thickness=mean_pixel_height)
    mmcq = preprocessed[-1]
    
    # Segment entire choroid and proper opening to remove artefacts
    K, N_keep = seg_params
    sl_kwargs = dict(cluster_type="mediancut", K=K, N_keep=N_keep)
    large_vessel_binmap = compute_vessel_segmentation(mmcq, shifted_bnds[...,1], **sl_kwargs)
    #large_foot = morph.footprints.disk(radius=3)
    #large_vessel_binmap = morphological_filter(large_vessel_binmap, footprint=large_foot, type="open")

    # Choriocapillaris is between 0 and 10 pixels, then we look for medium vessels based on mean thickness
    upper_vessel_binmap = np.zeros_like(large_vessel_binmap)
    for delta in [cc_delta, small_delta, mean_pixel_height//3]:
        off_bnds = taper_duplicated_upper(shifted_bnds, delta=delta)
        sl_kwargs["N_keep"] = N_keep
        if delta == cc_delta:
            for (x,y) in shifted_bnds[0]:
                upper_vessel_binmap[y:y+cc_delta, x] = 1
            all_vbinmaps.append(upper_vessel_binmap.copy())
        else: 
            delta_vessel_binmap = compute_vessel_segmentation(mmcq, off_bnds[...,1], **sl_kwargs)
            delta_foot = morph.footprints.disk(radius=2)
            
            upper_vessel_binmap += morphological_filter(delta_vessel_binmap, footprint=delta_foot, type="open")
    all_vbinmaps.append(upper_vessel_binmap.copy())

    # Compute vessel binary map for Sattler's layer
    if mean_pixel_height//3 > small_delta:
        offset_l = -3*mean_pixel_height//5
        medium_bnds = offset_trace(shifted_bnds, small_delta//2, offset_l)
        sl_kwargs["stx"] = medium_bnds[0,0,0]
        medium_vessel_binmap = compute_vessel_segmentation(mmcq, medium_bnds[...,1], **sl_kwargs)
        medium_foot = morph.footprints.disk(radius=2)
        medium_vessel_binmap = morphological_filter(medium_vessel_binmap, footprint=medium_foot, type="open")
    else:
        logging.warning("Choroid too small to individually segment Sattler's layer.")
        medium_vessel_binmap = large_vessel_binmap.copy()
    
    # Combine binmaps
    vessel_binmap = (large_vessel_binmap + medium_vessel_binmap + upper_vessel_binmap).clip(0, 1).astype(int)

    # Collect binary maps at different depths
    all_vbinmaps += [medium_vessel_binmap, large_vessel_binmap]
    all_vbinmaps = [rebuild_map(m, img.shape, crop_idx) for m in all_vbinmaps]

    # Final vessel mask is rebuilt back to original dimensions
    vmask = rebuild_map(vessel_binmap, img.shape, crop_idx)
    output = vmask
    if return_preprocessed:
        output = preprocessed, all_vbinmaps, vmask

    return output



def niblack(img, bnds, W=51, k=-0.05, preprocess=True):
    """
    Apply Niblack auto-local thresholding using window_size=15 and k is
    standard deviation weight to adjust how much of the pixel intensity
    distribution empirically observed in the window is taken as part of 
    the local thresholding calculation. 

    Note, if k << 0 then threshold will accept everything. If k >> 0
    then threshold will reject everything. Typically -0.01 > k > 0.01.
    Note: Larger window should have a negative k, so as to remove
    outliers influencing threshold. Smaller windows should have a larger k.

    This paper: https://www.mdpi.com/2304-6732/10/3/234
    fixed parameters for different retinal disease
    W = 13 and k = 0.01
    W = 15 and k = 0.001

    This paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8857621/
    W = 20 --- 75 and k = -0.05
    """

    # 1) Preprocess by removing unwanted area, and preprocess
    bnds = crop_trace(interp_trace(bnds))
    roi, shifted_bnds, crop_idx = crop_ROI(img, bnds)
    if preprocess:
        nib_prep = preprocess_vesselseg(roi, mmcq=False)[-1]
    else:
        nib_prep = roi.copy()
    
    # 2) Apply Niblack 
    T = threshold_niblack(nib_prep, window_size=W, k=k)
    nib_img = (nib_prep > T).astype(int)
    
    # 3) Rebuild to output binary mask
    bin_pixels, bin_idx = get_features(nib_img, shifted_bnds[...,1], stx=shifted_bnds[...,0].min(), norm=False)
    vessel_pixels = bin_idx[bin_pixels == 0]
    nib_bmap = np.zeros_like(nib_prep)
    nib_bmap[vessel_pixels[:,1], vessel_pixels[:,0]] = 1    
    
    return rebuild_map(nib_bmap, img.shape, crop_idx)


################################################################################################
#                       ROBUST VESSEL SEGMENTATION
################################################################################################

# Contrast levels fixed between 0.5 and 3

def vary_contrast(img, factor):
    """Wrapper function for varying contrast via Torchvision"""
    return F.adjust_contrast(torch.tensor(img).unsqueeze(0).unsqueeze(0), contrast_factor=factor).squeeze(0).squeeze(0).numpy()

def get_coef_range(img, mask, bounds=(0.2, 0.5), N=5):
    """Function to determine what Gamma level to enhance image"""
    upper, lower = bounds
    if isinstance(mask, tuple):
        mask = rebuild_mask(mask, img.shape)
    img_mean = (img[mask.astype(bool)]).mean()
    return np.linspace(np.log(upper) / np.log(img_mean), np.log(lower) / np.log(img_mean), N)

def robust_vessel_seg(img, bnds, seg_func, **kwargs):
    """
    Segment choroidal vessels after varying brightness and contrast at 5 different levels, using 3:2 majority vote.
    """
    # Loop over gamma and contrast levels, majority vote is done per level 
    cg_seg = np.zeros_like(img)
    gamma_coeffs = get_coef_range(img, bnds)
    con_lvls = np.linspace(0.5, 3, 5)
    for g in gamma_coeffs:
        #print(f"     {g}")
        gam_img = exposure.adjust_gamma(img, gamma=g)
        c_seg = np.zeros_like(img)
        for c in con_lvls:
            gam_con_img = vary_contrast(gam_img, c)
            try:
                #print(f"        {c}")
                c_seg += seg_func(gam_con_img, bnds, **kwargs)
            except:
                #print(f"        Failed at {c}")
                c_seg += np.zeros_like(c_seg)
        cg_seg += (c_seg > 2).astype(int)
    robust_seg = (cg_seg > 2).astype(int)
    
    return robust_seg
        




