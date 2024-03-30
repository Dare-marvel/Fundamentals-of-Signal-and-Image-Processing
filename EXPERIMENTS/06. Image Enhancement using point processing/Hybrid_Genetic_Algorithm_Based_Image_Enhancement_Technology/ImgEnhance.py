import numpy as np
import scipy
from skimage import io, color, img_as_float
from skimage.transform import resize
from scipy.optimize import differential_evolution
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import peak_signal_noise_ratio
import matplotlib.pyplot as plt

def beta_function(u, alpha, beta):
    """
    Normalized incomplete Beta function for image enhancement.
    """
    x = u.copy()
    out = np.zeros_like(x)
    mask = (x >= 0) & (x <= 1)  # Create a mask to handle values outside [0, 1]
    x = x[mask]  # Apply the mask
    out[mask] = x ** alpha * (1 - x) ** beta / scipy.special.beta(alpha, beta)
    return out

def enhance_image(image, bounds):
    """
    Enhance the image using the Hybrid Genetic Algorithm.
    """
    def objective_func(params):
        alpha, beta = params
        enhanced = beta_function(image, alpha, beta)
        # Specify the data range for SSIM
        return -np.mean(ssim(image, enhanced, data_range=image.max() - image.min()))

    result = differential_evolution(objective_func, bounds, maxiter=600, popsize=30, disp=False, workers=1)
    alpha, beta = result.x
    enhanced = beta_function(image, alpha, beta)
    return enhanced

def safe_psnr(img1, img2, data_range=None):
    """
    Compute the peak signal-to-noise ratio (PSNR) between two images, avoiding division by zero.
    """
    if data_range is None:
        data_range = np.max(img1) - np.min(img1)
    err = np.mean((img1 - img2) ** 2)
    return 10 * np.log10((data_range ** 2) / (err + 1e-12))  # Add a small constant to avoid division by zero

def main():
    # Load the image
    original_image = io.imread('C:\\Users\\aspur\\OneDrive\\FOSIP\\EXPERIMENTS\\06. Image Enhancement using point processing\\input_image.png')

    # Check if image has four channels (RGBA)
    if original_image.shape[2] == 4:
        # Use only RGB channels
        original_image = original_image[:, :, :3]

    # Convert to grayscale
    original_image_gray = color.rgb2gray(original_image)

    # Enhance the image
    bounds = [(1, 20), (1, 20)]
    enhanced = enhance_image(original_image_gray, bounds)

    # Convert enhanced image mode to a supported mode for PNG
    enhanced = color.gray2rgb(enhanced)

    # Ensure images have the same dimensions
    if original_image.shape != enhanced.shape:
        # If dimensions don't match, pad or crop the enhanced image
        if original_image.shape[0] < enhanced.shape[0]:
            enhanced = enhanced[:original_image.shape[0], :original_image.shape[1], :]
        elif original_image.shape[0] > enhanced.shape[0]:
            pad_width = ((0, original_image.shape[0] - enhanced.shape[0]), (0, original_image.shape[1] - enhanced.shape[1]), (0, 0))
            enhanced = np.pad(enhanced, pad_width, mode='constant')

    # Convert the images to the same data type
    original_image = img_as_float(original_image)
    enhanced = img_as_float(enhanced)

    # Check if the original and enhanced images are identical
    if np.array_equal(original_image, enhanced):
        print("Original and enhanced images are identical.")
        original_psnr = 0
        enhanced_psnr = 0
    else:
        # Evaluate the performance
        original_psnr = safe_psnr(original_image, original_image)
        enhanced_psnr = safe_psnr(original_image, enhanced)

    print(f"Original PSNR: {original_psnr:.2f}")
    print(f"Enhanced PSNR: {enhanced_psnr:.2f}")

    # Save the enhanced image
    enhanced_uint8 = (enhanced * 255).astype(np.uint8)
    enhanced_uint8 = np.squeeze(enhanced_uint8)  # Remove single-dimensional entries
    io.imsave('C:\\Users\\aspur\\OneDrive\\FOSIP\\EXPERIMENTS\\06. Image Enhancement using point processing\\enhanced_image.png', enhanced_uint8)

    # Plot histograms
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(original_image.ravel(), bins=256, color='blue', alpha=0.7)
    plt.title('Original Image Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.subplot(1, 2, 2)
    plt.hist(enhanced.ravel(), bins=256, color='red', alpha=0.7)
    plt.title('Enhanced Image Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.show()

if __name__ == "__main__":
    main()