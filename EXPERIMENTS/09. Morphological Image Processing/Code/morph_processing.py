import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from scipy.ndimage import grey_opening, grey_closing
from pywt import dwt2, idwt2
from sklearn.decomposition import PCA

def morph_unsharp_mask(img):
    """
    Applies morphological operations based unsharp masking to enhance the image.
    """
    # Gaussian blurring
    blurred = gaussian_filter(img, sigma=1)
    
    # Morphological opening
    opened = grey_opening(img, size=(3, 3))
    
    # Morphological closing
    closed = grey_closing(opened, size=(3, 3))
    
    # Unsharp masking
    sharpened = img - closed
    enhanced = img + sharpened
    
    return enhanced

def curvelet_transform(img):
    """
    Applies curvelet transform to decompose the image into approximation and detailed coefficients.
    """
    cA, (cH, cV, cD) = dwt2(img, 'db1')
    return cA, (cH, cV, cD)

def fusion_approx(cA_ir, cA_vi):
    """
    Fuses the approximation coefficients using PCA.
    """
    X = np.stack([cA_ir.ravel(), cA_vi.ravel()], axis=1)
    pca = PCA(n_components=1)
    cA_fused = pca.fit_transform(X).reshape(cA_ir.shape)
    return cA_fused

def fusion_detail(cH_ir, cV_ir, cD_ir, cH_vi, cV_vi, cD_vi):
    """
    Fuses the detailed coefficients using the max rule.
    """
    cH_fused = np.maximum(cH_ir, cH_vi)
    cV_fused = np.maximum(cV_ir, cV_vi)
    cD_fused = np.maximum(cD_ir, cD_vi)
    return cH_fused, cV_fused, cD_fused

def fuse_images(ir_img, vi_img):
    """
    Fuses the infrared and visible images using the proposed method.
    """
    # Resize the images to the same shape
    ir_img = cv2.resize(ir_img, vi_img.shape[:2][::-1])

    # Enhance the source images
    ir_enhanced = morph_unsharp_mask(ir_img)
    vi_enhanced = morph_unsharp_mask(vi_img)
    
    # Apply curvelet transform
    cA_ir, (cH_ir, cV_ir, cD_ir) = curvelet_transform(ir_enhanced)
    cA_vi, (cH_vi, cV_vi, cD_vi) = curvelet_transform(vi_enhanced)
    
    # Fuse the approximation and detailed coefficients
    cA_fused = fusion_approx(cA_ir, cA_vi)
    cH_fused, cV_fused, cD_fused = fusion_detail(cH_ir, cV_ir, cD_ir, cH_vi, cV_vi, cD_vi)
    
    # Reconstruct the fused image
    fused_img = idwt2((cA_fused, (cH_fused, cV_fused, cD_fused)), 'db1')
    
    return fused_img

def dwt_sharpen_fusion(ir_img, vi_img):
    """
    DWT and Sharpen filter based fusion.
    """
    # Resize the images to the same shape
    ir_img = cv2.resize(ir_img, vi_img.shape[:2][::-1])
    
    # Apply DWT and sharpen filter
    cA_ir, (cH_ir, cV_ir, cD_ir) = dwt2(ir_img, 'db1')
    cA_vi, (cH_vi, cV_vi, cD_vi) = dwt2(vi_img, 'db1')
    
    cA_fused = (cA_ir + cA_vi) / 2
    cH_fused = np.maximum(cH_ir, cH_vi)
    cV_fused = np.maximum(cV_ir, cV_vi)
    cD_fused = np.maximum(cD_ir, cD_vi)
    
    dwt_sharpen_result = idwt2((cA_fused, (cH_fused, cV_fused, cD_fused)), 'db1')
    return dwt_sharpen_result

def pca_multimodal_fusion(ir_img, vi_img):
    """
    PCA based multimodal fusion.
    """
    # Resize the images to the same shape
    ir_img = cv2.resize(ir_img, vi_img.shape[:2][::-1])
    
    # Apply PCA
    X = np.stack([ir_img.ravel(), vi_img.ravel()], axis=1)
    pca = PCA(n_components=1)
    pca_multimodal_result = pca.fit_transform(X).reshape(ir_img.shape)
    return pca_multimodal_result

def curvelet_fusion(ir_img, vi_img):
    """
    Curvelet multi-scale transform based fusion.
    """
    # Resize the images to the same shape
    ir_img = cv2.resize(ir_img, vi_img.shape[:2][::-1])
    
    # Apply curvelet transform
    cA_ir, (cH_ir, cV_ir, cD_ir) = curvelet_transform(ir_img)
    cA_vi, (cH_vi, cV_vi, cD_vi) = curvelet_transform(vi_img)
    
    cA_fused = fusion_approx(cA_ir, cA_vi)
    cH_fused, cV_fused, cD_fused = fusion_detail(cH_ir, cV_ir, cD_ir, cH_vi, cV_vi, cD_vi)
    
    curvelet_result = idwt2((cA_fused, (cH_fused, cV_fused, cD_fused)), 'db1')
    return curvelet_result

def dwt_unsharp_fusion(ir_img, vi_img):
    """
    DWT and unsharp masking based fusion.
    """
    # Resize the images to the same shape
    ir_img = cv2.resize(ir_img, vi_img.shape[:2][::-1])
    
    # Apply DWT and unsharp masking
    ir_enhanced = morph_unsharp_mask(ir_img)
    vi_enhanced = morph_unsharp_mask(vi_img)
    
    cA_ir, (cH_ir, cV_ir, cD_ir) = dwt2(ir_enhanced, 'db1')
    cA_vi, (cH_vi, cV_vi, cD_vi) = dwt2(vi_enhanced, 'db1')
    
    cA_fused = fusion_approx(cA_ir, cA_vi)
    cH_fused, cV_fused, cD_fused = fusion_detail(cH_ir, cV_ir, cD_ir, cH_vi, cV_vi, cD_vi)
    
    dwt_unsharp_result = idwt2((cA_fused, (cH_fused, cV_fused, cD_fused)), 'db1')
    return dwt_unsharp_result

# Load the input images
ir_img = cv2.imread('B:\\morph_processing\\Input_Images\\stairs\\stairs_ir.png', cv2.IMREAD_GRAYSCALE)
vi_img = cv2.imread('B:\\morph_processing\\Input_Images\\stairs\\stairs_vi.png', cv2.IMREAD_GRAYSCALE)

# Fuse the images using the different methods
dwt_sharpen_result = dwt_sharpen_fusion(ir_img, vi_img)
pca_multimodal_result = pca_multimodal_fusion(ir_img, vi_img)
curvelet_result = curvelet_fusion(ir_img, vi_img)
dwt_unsharp_result = dwt_unsharp_fusion(ir_img, vi_img)
proposed_result = fuse_images(ir_img, vi_img)

# Save the fused images
cv2.imwrite('dwt_sharpen_result.png', dwt_sharpen_result)
cv2.imwrite('pca_multimodal_result.png', pca_multimodal_result)
cv2.imwrite('curvelet_result.png', curvelet_result)
cv2.imwrite('dwt_unsharp_result.png', dwt_unsharp_result)
cv2.imwrite('proposed_result.png', proposed_result)