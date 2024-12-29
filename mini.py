import streamlit as st
import cv2
import numpy as np
from skimage.feature import hog
from skimage.filters import sobel, prewitt
from skimage.restoration import denoise_bilateral
from scipy.ndimage import gaussian_laplace
import matplotlib.pyplot as plt
from PIL import Image

############################
# Feature Analysis Function
############################

@st.cache_data
def process_all_features(img):
    results = []

    # Example: SIFT
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    img_sift = cv2.drawKeypoints(img, keypoints, None)
    results.append({"title": "SIFT", "image": img_sift, "desc": "SIFT keypoints and descriptors."})

    # Example: HOG
    fd, hog_image = hog(img, visualize=True)
    results.append({"title": "HOG", "image": hog_image, "desc": "Histogram of Oriented Gradients."})

    # Example: Sobel
    sobel_filtered = sobel(img)
    results.append({"title": "Sobel", "image": sobel_filtered, "desc": "Sobel filter."})

    # Example: Gaussian Laplace
    gauss_laplace_img = gaussian_laplace(img, sigma=1)
    results.append({"title": "Gaussian Laplace", "image": gauss_laplace_img, "desc": "Gaussian Laplace filter."})

    # Example: Bilateral
    img_bilateral = denoise_bilateral(img, sigma_color=0.05, sigma_spatial=15)
    results.append({"title": "Bilateral", "image": img_bilateral, "desc": "Bilateral filter."})

    return results

############################
# Streamlit App
############################

st.title("Feature Analysis and Filters")

# File Uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image using OpenCV
    img = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    img = np.array(img)

    st.image(img, caption="Original Image", use_column_width=True)

    # Process all features
    results = process_all_features(img)

    # Display results
    for result in results:
        st.subheader(result["title"])
        st.write(result["desc"])
        
        # Display image results
        if isinstance(result["image"], np.ndarray):
            plt.figure()
            plt.imshow(result["image"], cmap="gray")
            plt.axis("off")
            st.pyplot(plt)
