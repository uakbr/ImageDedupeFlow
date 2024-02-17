import os
from itertools import combinations
import concurrent.futures
from skimage import color, io, filters
from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2
from scipy.stats import chisquare

# Configure the thresholds for the comparison steps
chi2_threshold = 0.05  # P-value threshold for the Chi-Squared test in histogram comparison
ssim_threshold = 0.5  # Threshold for SSIM comparison
high_ssim_threshold = 0.95  # Threshold for high SSIM indicating very similar images
edge_difference_threshold = 10  # Threshold for the sum of absolute differences in edge detection
feature_match_threshold = 10  # Threshold for the number of feature matches

def histogram_comparison_chi2(image1, image2):
    gray1 = color.rgb2gray(image1)
    gray2 = color.rgb2gray(image2)
    hist1, _ = np.histogram(gray1.ravel(), bins=256, range=(0, 1))
    hist2, _ = np.histogram(gray2.ravel(), bins=256, range=(0, 1))
    chi2_stat, p_value = chisquare(hist1, f_exp=hist2)
    return p_value > chi2_threshold

def compute_ssim(image1, image2):
    try:
        gray1 = color.rgb2gray(image1)
        gray2 = color.rgb2gray(image2)
        ssim_value = ssim(gray1, gray2)
        return ssim_value
    except ValueError as e:
        print(f"Error computing SSIM: {e}")
        return 0  # Or handle the error as appropriate

def detect_and_describe_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

def match_features(desc1, desc2):
    matcher = cv2.BFMatcher()
    raw_matches = matcher.knnMatch(desc1, desc2, k=2)
    matches = [m for m, n in raw_matches if m.distance < 0.75 * n.distance]
    return matches

def edge_detection_comparison(image1, image2):
    edges1 = filters.sobel(color.rgb2gray(image1))
    edges2 = filters.sobel(color.rgb2gray(image2))
    edge_diff = np.abs(edges1 - edges2)
    return np.sum(edge_diff)

def image_comparison_pipeline(filepath1, filepath2):
    image1 = io.imread(filepath1)
    image2 = io.imread(filepath2)

    # Stage 1: Histogram Comparison
    if not histogram_comparison_chi2(image1, image2):
        return "Not Similar - Histograms differ significantly."

    # Stage 2: Structural Similarity Index
    ssim_value = compute_ssim(image1, image2)
    if ssim_value < ssim_threshold:
        return "Not Similar - Structural differences."
    elif ssim_value > high_ssim_threshold:
        return "Very Similar - High structural similarity."

    # Stage 3: Edge Detection and Matching
    edge_difference = edge_detection_comparison(image1, image2)
    if edge_difference > edge_difference_threshold:
        return "Different - Edge differences."

    # Stage 4: Feature Matching
    keypoints1, desc1 = detect_and_describe_features(image1)
    keypoints2, desc2 = detect_and_describe_features(image2)
    matches = match_features(desc1, desc2)
    if len(matches) > feature_match_threshold:
        return "Different - Significant feature matches."

    return "Similar - Passed all checks."

def list_images_in_folder(folder_path):
    """List full paths of images in a given folder."""
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(supported_formats)]

def compare_image_pair(pair):
    """Wrapper function to compare a pair of images."""
    img1, img2 = pair
    result = image_comparison_pipeline(img1, img2)
    return (img1, img2, result)

def compare_all_images_in_folder_parallel(folder_path):
    """Compare all images in a folder using parallel processing."""
    images = list_images_in_folder(folder_path)
    image_pairs = list(combinations(images, 2))  # Generate all possible pairs without repeating

    # Use ThreadPoolExecutor to parallelize image comparison
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_pair = {executor.submit(compare_image_pair, pair): pair for pair in image_pairs}
        for future in concurrent.futures.as_completed(future_to_pair):
            pair = future_to_pair[future]
            try:
                img1, img2, result = future.result()
                print(f"Comparison result for:\n{img1}\nand\n{img2}:\n{result}\n")
            except Exception as exc:
                print(f"Generated an exception: {pair}: {exc}")

# Usage
folder_path = 'C:\\file_name_here\\'
compare_all_images_in_folder_parallel(folder_path)
