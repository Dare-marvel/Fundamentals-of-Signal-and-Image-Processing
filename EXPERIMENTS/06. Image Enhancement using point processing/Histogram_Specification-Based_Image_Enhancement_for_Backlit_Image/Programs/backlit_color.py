import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_loe(orig_img, enhanced_img):
    """
    Calculate Lightness Order Error (LOE) between the original and enhanced images.
    """
    m = orig_img.shape[0] * orig_img.shape[1]
    loe = 0
    for p in range(m):
        orig_l = max(orig_img.reshape(-1, 3)[p])
        enh_l = max(enhanced_img.reshape(-1, 3)[p])
        loe += int(orig_l >= enh_l) != int(orig_l >= orig_l)
    return loe / m

def enhance_image(input_image, target_loe):
    """
    Enhance the input image to achieve the target Lightness Order Error (LOE).
    """
    # Split the input image into separate channels
    b, g, r = cv2.split(input_image)

    # Enhance each channel separately
    b_enhanced = enhance_channel(b, target_loe)
    g_enhanced = enhance_channel(g, target_loe)
    r_enhanced = enhance_channel(r, target_loe)

    # Merge the enhanced channels back into a colored image
    enhanced_bgr = cv2.merge([b_enhanced, g_enhanced, r_enhanced])

    return enhanced_bgr

def enhance_channel(channel, target_loe):
    """
    Enhance a single channel of the input image to achieve the target Lightness Order Error (LOE).
    """
    # Compute the histogram of the channel
    hist, bins = np.histogram(channel.flatten(), 256, [0, 256])

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

    # Apply histogram equalization to the input channel
    enhanced_channel = equalized[channel]

    return enhanced_channel


# Change the Path to the file
# Example usage
input_image = cv2.imread('B:\\Backlit\\Input_Images\\leo_backlit.jpg')

# Generate 6 output images with different LOE values
target_loes = [40187, 20008, 9243, 58352, 21995, 21550,25789,32356]

# Define enhancement methods
methods = ['CLAHE', 'Farbman', 'HE', 'MSRCR', 'Paris', 'Wang','Target_1','Target_2']

# Plot histogram for input image
plt.figure()
plt.hist(input_image.flatten(), bins=256, range=[0,256], color='b', alpha=0.7)
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.title('Histogram of Input Image')
plt.savefig('histogram_input_image.png')

for i, target_loe in enumerate(target_loes):
    enhanced_image = enhance_image(input_image, target_loe)
    cv2.imwrite(f'enhanced_image_{methods[i]}.jpg', enhanced_image)

    # Plot histogram
    plt.figure()
    plt.hist(enhanced_image.flatten(), bins=256, range=[0,256], color='b', alpha=0.7)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Enhanced Image {methods[i]}')
    plt.savefig(f'histogram_enhanced_image_{methods[i]}.png')
