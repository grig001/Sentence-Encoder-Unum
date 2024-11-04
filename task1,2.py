import os
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import local_binary_pattern
from sklearn.decomposition import PCA

old_stdout = sys.stdout
log_file = open("task1,2_output.log", "w", encoding='utf-8')
sys.stdout = log_file


def load_image(path):
    return cv2.imread(path)

def apply_pca(descriptors, n_components=32):
    # Apply PCA to reduce descriptor dimensionality
    pca = PCA(n_components=n_components)
    pca_descriptors = pca.fit_transform(descriptors)
    return pca_descriptors

def detect_keypoints_and_descriptors(image, method="SIFT"):
    
    if method == "SIFT":
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)

    elif method == "ORB":
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(image, None)

    elif method == "Color-SIFT":
        # Apply SIFT on each color channel
        channels = cv2.split(image)
        keypoints = []
        descriptors = []
        for channel in channels:
            sift = cv2.SIFT_create()
            kp, desc = sift.detectAndCompute(channel, None)
            keypoints.extend(kp)
            if desc is not None:
                descriptors.append(desc)
        descriptors = np.vstack(descriptors) if descriptors else None

    elif method == "PCA-SIFT":
        # Apply SIFT and then PCA to reduce descriptor dimensionality
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)
        if descriptors is not None:
            descriptors = apply_pca(descriptors, n_components=32)  # Reduce to 32 dimensions

    else:
        raise ValueError("Invalid method. Choose 'SIFT', 'ORB', 'Color-SIFT', or 'PCA-SIFT'.")
    
    return keypoints, descriptors

def show_keypoints(image, keypoints, method):
    # Draw keypoints on the image for visualization
    img_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite("src/week4/method_2/" + method + "_00000" + ".jpg", img_with_keypoints)

# Main processing function
def process_image(image_path, method="SIFT"):
    # Load image
    image = load_image(image_path)
    # Detect keypoints and compute descriptors
    keypoints, descriptors = detect_keypoints_and_descriptors(image, method)
    # Display keypoints
    # show_keypoints(image, keypoints, method)
    # print(f"Method: {method}, Keypoints: {len(keypoints)}, Descriptor shape: {descriptors.shape if descriptors is not None else 'None'}")


# Function to find tentative matches using different similarity metrics
def match_keypoints(descriptors1, descriptors2, ratio_test=0.75, method="L2"):
    # Set up matcher based on similarity metric
    if method == "L2":
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    elif method == "Hamming":
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        raise ValueError("Invalid similarity metric. Choose 'L2' or 'Hamming'.")

    # Find matches and apply Lowe's ratio test to filter matches
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_test * n.distance:
            good_matches.append(m)

    return good_matches

# Visualize matches between two images
def visualize_matches(image1, image2, keypoints1, keypoints2, matches, method):
    match_img = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    cv2.imwrite(f"src/week4/method_2/matches_{method}.jpg", match_img)
    plt.imshow(match_img)
    plt.axis('off')
    plt.show()

# Main function to perform detection, descriptor extraction, and matching
def process_and_match(image_path1, image_path2, method="ORB", metric="L2"):

    image1 = load_image(image_path1)
    image2 = load_image(image_path2)

    # Detect keypoints and descriptors for both images
    keypoints1, descriptors1 = detect_keypoints_and_descriptors(image1, method)
    keypoints2, descriptors2 = detect_keypoints_and_descriptors(image2, method)

    # Find tentative matches
    matches = match_keypoints(descriptors1, descriptors2, method=metric)

    # visualize_matches(image1, image2, keypoints1, keypoints2, matches, method)

    # print(f"Method: {method}, Metric: {metric}, Matches: {len(matches)}")
    return len(matches)


# Run on a sample image
image_path1 = 'datasets/qsd1_w4/00002.jpg'  
image_path2 = 'datasets/qsd1_w4/00006.jpg'

process_image(image_path1, method="SIFT")
process_image(image_path1, method="ORB")
process_image(image_path1, method="Color-SIFT")
process_image(image_path1, method="PCA-SIFT")

folder_path = 'datasets/qsd1_w4/'
matches = []
for filename_1 in os.listdir(folder_path):
    image_matches = []    
    for filename_2 in os.listdir(folder_path):
        if filename_1.lower().endswith(('.jpg')) and filename_2.lower().endswith(('.jpg')):
            if filename_1 == filename_2:
                image_matches.append(1e5)
            else:
                image_matches.append(process_and_match(folder_path + filename_1, folder_path + filename_2, method="ORB", metric="L2"))

    if len(image_matches) > 0:
        print(filename_1)
        print(image_matches)
        print('______________________________________________________________')

        matches.append(image_matches)


print(matches)

# process_and_match(image_path1, image_path2, method="SIFT", metric="L2")
# process_and_match(image_path1, image_path2, method="ORB", metric="Hamming")
# process_and_match(image_path1, image_path2, method="PCA-SIFT", metric="L2")


sys.stdout = old_stdout
log_file.close()