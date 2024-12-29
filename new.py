import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from skimage.feature import hog
from skimage.filters import gabor, sobel, prewitt, roberts
from skimage import exposure, measure
from scipy.ndimage import gaussian_laplace, laplace
from skimage.restoration import denoise_bilateral
from skimage.segmentation import watershed
from scipy.ndimage import label, distance_transform_edt
from skimage.color import label2rgb
from sklearn.metrics import f1_score, jaccard_score
from skimage.feature import canny
from io import BytesIO


# Function to process the uploaded image
def process_image(img):
    if img is None:
        return None, None, None

    # 1. SIFT Keypoints
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    img_sift = cv2.drawKeypoints(img, keypoints, None)

    # 2. HOG (Histogram of Oriented Gradients)
    fd, hog_image = hog(img, visualize=True)

    # 3. Gabor Filters
    gabor_filtered, _ = gabor(img, frequency=0.6)

    # 4. Gaussian Laplace
    gauss_laplace_img = gaussian_laplace(img, sigma=1)

    # 5. Laplace Filter
    laplace_filtered = laplace(img)

    # 6. Watershed Segmentation
    distance = distance_transform_edt(img)
    local_maxi = measure.label(distance)
    labels_ws = watershed(-distance, local_maxi, mask=img)
    label_img = label2rgb(labels_ws, image=img)

    # 7. Bilateral Filtering
    img_bilateral = denoise_bilateral(img, sigma_color=0.05, sigma_spatial=15)

    # 8. Gaussian Blurring
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)

    # 9. Median Filtering
    median_filtered = cv2.medianBlur(img, 5)

    # 10. Sobel Filtering
    sobel_filtered = sobel(img)

    # 11. Prewitt Filter
    prewitt_filtered = prewitt(img)

    # 12. Roberts Filter
    roberts_filtered = roberts(img)

    # 13. MSER (Maximally Stable Extremal Regions) Detection
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(img)
    img_mser = np.copy(img)
    for region in regions:
        hull = cv2.convexHull(np.array(region).reshape(-1, 1, 2))
        cv2.polylines(img_mser, [hull], 1, (255, 0, 0), 1)

    return (img_sift, hog_image, gabor_filtered, gauss_laplace_img, laplace_filtered,
            label_img, img_bilateral, blurred_img, median_filtered, sobel_filtered,
            prewitt_filtered, roberts_filtered, img_mser)


# Streamlit Application
st.title('Image Processing Pipeline with Drag and Drop')
st.write("Upload an image using the drag-and-drop feature to see various image processing techniques.")

# Upload image using file uploader
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded image to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)  # Read the image as color image (BGR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for processing

    # Display the uploaded image
    st.subheader('Uploaded Image')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Process the image
    results = process_image(img_gray)

    # Display results for various image processing techniques
    technique_titles = [
        "SIFT Keypoints", "HOG (Histogram of Oriented Gradients)", "Gabor Filter", 
        "Gaussian Laplace", "Laplace Filter", "Watershed Segmentation", 
        "Bilateral Filter", "Gaussian Blurring", "Median Filtering", 
        "Sobel Filtering", "Prewitt Filtering", "Roberts Filtering", 
        "MSER Object Detection"
    ]
    
    for i, result in enumerate(results):
        st.subheader(technique_titles[i])
        st.image(result, caption=technique_titles[i], use_column_width=True)

    # Accuracy Metrics (example: F1-Score for Sobel Edge Detection and IoU for Watershed)
    st.subheader('Accuracy Metrics')

    # 1. F1 Score for Edge Detection (Sobel)
    ground_truth = canny(img_gray)
    sobel_edges = sobel(img_gray)
    sobel_edges_binary = sobel_edges > 0.1  # Thresholding to binary

    f1 = f1_score(ground_truth.flatten(), sobel_edges_binary.flatten())
    st.write(f'F1 Score for Sobel Edge Detection: {f1:.4f}')

    # 2. Jaccard Index (IoU) for Watershed Segmentation
    ground_truth_mask = np.random.randint(0, 2, size=img_gray.shape)  # Dummy mask (replace with actual)
    iou = jaccard_score(ground_truth_mask.flatten(), results[5].flatten())
    st.write(f'Jaccard Index (IoU) for Watershed Segmentation: {iou:.4f}')

    # 3. Dice Coefficient for Watershed Segmentation
    dice = 2 * np.sum(ground_truth_mask * results[5]) / (np.sum(ground_truth_mask) + np.sum(results[5]))
    st.write(f'Dice Coefficient for Watershed Segmentation: {dice:.4f}')
else:
    st.info("Please upload an image to begin.")
