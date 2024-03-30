import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt

def lda_threshold(hist):
    """
    Calculate the threshold using Linear Discriminant Analysis (LDA) for
    separating the foreground (backlit) and background (frontlit) regions.
    """
    tot = hist.sum()
    weights = hist / tot
    means = np.arange(256) * weights
    mean_total = means.sum()
    mean_squares = np.arange(256) ** 2 * weights
    mean_sq_total = mean_squares.sum()
    variance_between = mean_total * (1 - mean_total)
    variances = mean_squares - means ** 2
    variance_within = variances.sum()
    means_diff = means - mean_total
    lda_scores = means_diff ** 2 / (variance_within + 1e-8)
    threshold = np.argmax(lda_scores)
    return threshold

def histogram_specification(img, target_hist):
    """
    Perform histogram specification to match the input image's histogram
    to the target histogram.
    """
    hist, bins = np.histogram(img.ravel(), 256, [0, 256])
    cdf = hist.cumsum() / hist.sum()
    target_cdf = target_hist.cumsum() / target_hist.sum()
    lut = np.interp(cdf, target_cdf, np.arange(256))
    result = np.interp(img.ravel(), np.arange(256), lut).reshape(img.shape)
    return result.astype(np.uint8)

def saturation_preservation(img, enhanced):
    """
    Preserve the saturation of the input image in the enhanced image.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)[:, :, 2]
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def backlit_enhancement(img, target_type='basic'):
    """
    Enhance a backlit image using histogram specification and saturation preservation.
    """
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_y = img_yuv[:, :, 0]
    hist, bins = np.histogram(img_y.ravel(), 256, [0, 256])
    threshold = lda_threshold(hist)

    if target_type == 'basic':
        target_hist = np.zeros(256, dtype=np.float32)
        target_hist[0:threshold] = 1 / threshold
        target_hist[threshold:256] = 1 / (256 - threshold)
    elif target_type == 'target1':
        target_hist = hist.copy().astype(np.float32)
        target_hist[0:threshold] /= target_hist[0:threshold].sum()
        target_hist[threshold:256] /= target_hist[threshold:256].sum()
    elif target_type == 'target2':
        target_hist = hist.copy().astype(np.float32)
        target_hist[threshold:256] /= target_hist[threshold:256].sum()
    else:
        raise ValueError("Invalid target histogram type")

    enhanced_y = histogram_specification(img_y, target_hist)
    enhanced_yuv = img_yuv.copy()
    enhanced_yuv[:, :, 0] = enhanced_y
    enhanced_bgr = cv2.cvtColor(enhanced_yuv, cv2.COLOR_YUV2BGR)
    enhanced_final = saturation_preservation(img, enhanced_bgr)
    return enhanced_final 

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Backlit Image Enhancement')
    parser.add_argument('input_image', help='Path to the input backlit image')
    parser.add_argument('--target', choices=['basic', 'target1', 'target2'], default='target1',
                        help='Type of target histogram (default: target1)')
    args = parser.parse_args()

    # Load the input image
    img = cv2.imread(args.input_image)

    # Check if the image was loaded successfully
    if img is None:
        print(f"Error: Could not load image '{args.input_image}'")
        return
    
    # Calculate histograms for original and enhanced images
    original_hist = cv2.calcHist([img], [0], None, [256], [0, 256])

    # Enhance the backlit image
    enhanced = backlit_enhancement(img, target_type=args.target)

    # Calculate histogram for the enhanced image
    enhanced_hist = cv2.calcHist([enhanced], [0], None, [256], [0, 256])

    # Enhance the backlit image
    enhanced = backlit_enhancement(img, target_type=args.target)

    # Display the original and enhanced images
    cv2.imshow('Original', img)
    cv2.imshow('Enhanced', enhanced)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    plt.plot(original_hist, color='blue')
    plt.plot(enhanced_hist, color='red')
    plt.title('Histogram Comparison')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend(['Original', 'Enhanced'])
    plt.show()

if __name__ == '__main__':
    main()