import cv2
import numpy as np

def calculate_loe(orig_img, enhanced_img):
    """
    Calculate Lightness Order Error (LOE) between the original and enhanced images.
    """
    m = orig_img.size
    loe = 0
    for p in range(m):
        for q in range(m):
            orig_l = max(orig_img.getpixel((p, q)))
            enh_l = max(enhanced_img.getpixel((p, q)))
            loe += int(orig_l >= enh_l) != int(orig_l >= orig_l)
    return loe / m**2

def enhance_image(input_image, target_loe):
    """
    Enhance the input image to achieve the target Lightness Order Error (LOE).
    """
    # Convert input image to grayscale
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Compute the histogram of the grayscale image
    hist, bins = np.histogram(gray.flatten(), 256, [0, 256])

    # Compute the cumulative distribution function (CDF)
    cdf = hist.cumsum()
    cdf = cdf / cdf[-1]  # Normalize the CDF

    # Compute the target CDF based on the target LOE
    target_cdf = np.linspace(0, 1, 256)
    if target_loe < 9243:
        # Use a triangular target CDF for low LOE values
        target_cdf = np.minimum(2 * target_cdf, 2 * (1 - target_cdf))
    elif target_loe < 21550:
        # Use a quadratic target CDF for moderate LOE values
        target_cdf = target_cdf ** 2
    else:
        # Use a different target CDF for high LOE values
        target_cdf = np.sqrt(target_cdf)

    # Compute the equalized pixel values using the target CDF
    equalized = np.interp(cdf, target_cdf, np.linspace(0, 255, 256)).astype(np.uint8)

    # Apply histogram equalization to the input image
    enhanced = equalized[gray]

    # Convert the enhanced grayscale image back to BGR
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    return enhanced_bgr

# Example usage
input_image = cv2.imread('B:\\Backlit\\leo_backlit.jpg')

# Generate 6 output images with different LOE values
target_loes = [40187, 20008, 9243, 58352, 21995, 5144]
for i, target_loe in enumerate(target_loes):
    enhanced_image = enhance_image(input_image, target_loe)
    cv2.imwrite(f'enhanced_image_{i+1}.jpg', enhanced_image)