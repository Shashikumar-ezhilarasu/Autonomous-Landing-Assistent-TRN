import streamlit as st
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.feature import hog
from skimage.filters import gabor, sobel, prewitt, roberts, laplace
from skimage import exposure, measure, feature, filters
from skimage.restoration import denoise_bilateral
from skimage.segmentation import watershed
from skimage.color import label2rgb
from skimage.morphology import erosion, dilation, opening, closing
from scipy import ndimage as ndi
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import gaussian_laplace

############################
# Feature/Filter Analysis (Independent Component)
############################

@st.cache_data
def process_all_features(img):
    results = []
    # 1. SIFT
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    img_sift = cv2.drawKeypoints(img, keypoints, None)
    results.append({"title": "SIFT", "image": img_sift, "desc": "SIFT keypoints and descriptors."})

    # 2. HOG
    fd, hog_image = hog(img, visualize=True)
    results.append({"title": "HOG", "image": hog_image, "desc": "Histogram of Oriented Gradients."})

    # 3. Gabor
    gabor_filtered, _ = gabor(img, frequency=0.6)
    results.append({"title": "Gabor", "image": gabor_filtered, "desc": "Gabor filter."})

    # 4. Gaussian Laplace
    gauss_laplace_img = gaussian_laplace(img, sigma=1)
    results.append({"title": "Gaussian Laplace", "image": gauss_laplace_img, "desc": "Gaussian Laplace filter."})

    # 5. Laplace
    laplace_filtered = laplace(img)
    results.append({"title": "Laplace", "image": laplace_filtered, "desc": "Laplace filter."})

    # 6. Sobel
    sobel_filtered = sobel(img)
    results.append({"title": "Sobel", "image": sobel_filtered, "desc": "Sobel filter."})

    # 7. Prewitt
    prewitt_filtered = prewitt(img)
    results.append({"title": "Prewitt", "image": prewitt_filtered, "desc": "Prewitt filter."})

    # 8. Roberts
    roberts_filtered = roberts(img)
    results.append({"title": "Roberts", "image": roberts_filtered, "desc": "Roberts filter."})

    # 9. Bilateral
    img_bilateral = denoise_bilateral(img, sigma_color=0.05, sigma_spatial=15)
    results.append({"title": "Bilateral", "image": img_bilateral, "desc": "Bilateral filter."})

    # 10. Gaussian Blur
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
    results.append({"title": "Gaussian Blur", "image": blurred_img, "desc": "Gaussian blur."})

    # 11. Median
    median_filtered = cv2.medianBlur(img, 5)
    results.append({"title": "Median", "image": median_filtered, "desc": "Median filter."})

    # 12. Dilation
    dilated_img = dilation(img)
    results.append({"title": "Dilation", "image": dilated_img, "desc": "Morphological dilation."})

    # 13. Erosion
    eroded_img = erosion(img)
    results.append({"title": "Erosion", "image": eroded_img, "desc": "Morphological erosion."})

    # 14. Opening
    opened_img = opening(img)
    results.append({"title": "Opening", "image": opened_img, "desc": "Morphological opening."})

    # 15. Closing
    closed_img = closing(img)
    results.append({"title": "Closing", "image": closed_img, "desc": "Morphological closing."})

    # 16. Canny
    canny_edges = cv2.Canny(img, 100, 200)
    results.append({"title": "Canny", "image": canny_edges, "desc": "Canny edge detection."})

    # 17. Harris Corner
    harris_corners = cv2.cornerHarris(img, 2, 3, 0.04)
    img_harris = img.copy()
    img_harris[harris_corners > 0.01 * harris_corners.max()] = 255
    results.append({"title": "Harris", "image": img_harris, "desc": "Harris corner detection."})

    # 18. MSER
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(img)
    img_mser = np.copy(img)
    for region in regions:
        hull = cv2.convexHull(np.array(region).reshape(-1, 1, 2))
        cv2.polylines(img_mser, [hull], 1, (255, 0, 0), 1)
    results.append({"title": "MSER", "image": img_mser, "desc": "Maximally Stable Extremal Regions."})

    # 19. Gaussian Noise
    noisy_img = img + np.random.normal(0, 25, img.shape)
    results.append({"title": "Gaussian Noise", "image": noisy_img, "desc": "Added Gaussian noise."})

    # 20. Adaptive Threshold
    adaptive_thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
    results.append({"title": "Adaptive Threshold", "image": adaptive_thresh, "desc": "Adaptive thresholding."})

    # 21. Otsu
    _, otsu_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results.append({"title": "Otsu", "image": otsu_thresh, "desc": "Otsu thresholding."})

    # 22. Fourier Transform
    f_transform = np.fft.fft2(img)
    f_transform_shifted = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted) + 1e-10)
    results.append({"title": "Fourier Transform", "image": magnitude_spectrum, "desc": "Magnitude spectrum."})

    # 23. Hist Equalization
    equalized_img = cv2.equalizeHist(img)
    results.append({"title": "Hist Equalization", "image": equalized_img, "desc": "Histogram Equalization."})

    # 24. CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(img)
    results.append({"title": "CLAHE", "image": clahe_img, "desc": "Contrast Limited Adaptive Hist Eq."})

    # 25. Hough Lines
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    img_hough = img.copy()
    if lines is not None:
        for rho, theta in lines[:, 0]:
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a*rho, b*rho
            x1, y1 = int(x0 + 1000*(-b)), int(y0 + 1000*(a))
            x2, y2 = int(x0 - 1000*(-b)), int(y0 - 1000*(a))
            cv2.line(img_hough, (x1, y1), (x2, y2), (255, 0, 0), 1)
    results.append({"title": "Hough Lines", "image": img_hough, "desc": "Detected lines with Hough transform."})

    # 26. Distance Transform
    dist_transform = distance_transform_edt(img)
    results.append({"title": "Distance Transform", "image": dist_transform, "desc": "Euclidean distance transform."})

    # 27. Watershed
    distance = distance_transform_edt(img)
    local_maxi = measure.label(distance)
    labels_ws = watershed(-distance, local_maxi, mask=img)
    label_img = label2rgb(labels_ws, image=img)
    results.append({"title": "Watershed", "image": label_img, "desc": "Watershed segmentation."})

    # 28. Scharr
    scharr_img = filters.scharr(img)
    results.append({"title": "Scharr", "image": scharr_img, "desc": "Scharr edge detection."})

    # 29. Blob (LoG)
    blobs_log = feature.blob_log(img, max_sigma=30, num_sigma=10, threshold=0.1)
    img_blob = np.copy(img)
    for blob in blobs_log:
        y, x, r = blob
        cv2.circle(img_blob, (int(x), int(y)), int(r), (255, 0, 0), 1)
    results.append({"title": "Blob Detection", "image": img_blob, "desc": "LoG-based blob detection."})

    # 30. Median Denoising
    img_denoised = filters.median(img)
    results.append({"title": "Median Denoising", "image": img_denoised, "desc": "Median filter denoising."})

    # 31. Laplacian of Gaussian
    img_log = gaussian_laplace(img, sigma=1)
    results.append({"title": "LoG", "image": img_log, "desc": "Laplacian of Gaussian."})

    # 32. Connected Components
    num_labels, labels_img = cv2.connectedComponents(img)
    results.append({"title": "Connected Components", "image": labels_img,
                    "desc": f"{num_labels} connected components."})

    # 33. Hough Circle Detection
    img_circles = cv2.medianBlur(img, 5)
    circles = cv2.HoughCircles(img_circles, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                               param1=50, param2=30, minRadius=1, maxRadius=30)
    img_hough_circles = np.copy(img)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for c in circles[0, :]:
            cv2.circle(img_hough_circles, (c[0], c[1]), c[2], (255, 0, 0), 1)
    results.append({"title": "Hough Circles", "image": img_hough_circles, "desc": "Detected circles with Hough."})

    # 34. Skeletonization
    skeleton = feature.canny(img)
    results.append({"title": "Skeleton", "image": skeleton, "desc": "Skeletonization with Canny edges."})

    # Additional features omitted for brevity...



    return results

def feature_analysis_component(img):
    features_data = process_all_features(img)
    for i in range(0, len(features_data), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            if i + j < len(features_data):
                data = features_data[i + j]
                fig, ax = plt.subplots()
                ax.imshow(data["image"], cmap='gray')
                ax.set_title(data["title"])
                ax.axis('off')
                col.pyplot(fig)
                col.write(data["desc"])
                plt.close(fig)

##########################
# Landing Safety Analysis (Independent Component)
##########################

import os
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Cache for performance
@st.cache_data
def calculate_surface_slope(image):
    """
    Calculate the surface slope using the Sobel operator.
    """
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    slope = np.sqrt(grad_x**2 + grad_y**2)
    return np.mean(slope)

@st.cache_data
def calculate_surface_roughness(image):
    """
    Calculate the surface roughness using the Laplacian operator.
    """
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    return np.var(laplacian)

@st.cache_data
def assess_landing_safety(image):
    """
    Assess landing safety based on slope and roughness thresholds.
    """
    slope = calculate_surface_slope(image)
    roughness = calculate_surface_roughness(image)
    max_slope = 15  # Maximum allowable slope
    max_roughness = 5  # Maximum allowable roughness
    
    if slope <= max_slope and roughness <= max_roughness:
        return "safe"
    else:
        return "unsafe"

def preprocess_image(image):
    """
    Preprocess the input image (e.g., noise removal, normalization, etc.).
    In this case, we apply Gaussian blur to reduce noise.
    """
    # Apply Gaussian blur to the grayscale image
    preprocessed_image = cv2.GaussianBlur(image, (5, 5), 0)
    return preprocessed_image

def landing_safety_component(data_dir, ground_truth_labels, num_images_to_plot):
    """
    Main component to assess landing safety for images and display results.
    """
    image_subfolder = 'msl/images/edr'
    image_files = sorted([
        f for f in os.listdir(os.path.join(data_dir, image_subfolder))
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    image_files = image_files[:num_images_to_plot]

    correct_predictions = 0
    total_images = len(image_files)
    safe_count = 0
    unsafe_count = 0

    for i, image_file in enumerate(image_files):
        image_path = os.path.join(data_dir, image_subfolder, image_file)
        image = cv2.imread(image_path)
        if image is None:
            st.write(f"Error reading file: {image_path}.")
            continue

        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        preprocessed_image = preprocess_image(gray)

        # Assess safety of landing
        safety_status = assess_landing_safety(preprocessed_image)
        if safety_status == "safe":
            safe_count += 1
        else:
            unsafe_count += 1

        # Display image and safety status
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        axs[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axs[0, 0].set_title(f"Safety Status: {safety_status}")
        axs[0, 0].axis('off')

        axs[1, 0].imshow(preprocessed_image, cmap='gray')
        axs[1, 0].set_title(f"Processed Image")
        axs[1, 0].axis('off')

        st.pyplot(fig)
        plt.close(fig)

        # Display ground truth if available
        if i < len(ground_truth_labels):
            st.write(f"Ground Truth: {ground_truth_labels[i]}")

    if total_images > 0:
        accuracy = (correct_predictions / total_images) * 100
        st.write(f"Model Accuracy: {accuracy:.2f}%")

        if safe_count > unsafe_count:
            st.subheader("Overall Conclusion: Safe to Land")
        else:
            st.subheader("Overall Conclusion: Not Safe")
###################
# Chatbot Part (Independent Component)
###################

def chatbot_reply(question: str) -> str:
    feature_descriptions = {
        "sift": "SIFT stands for Scale-Invariant Feature Transform. It detects and describes local features in images.",
        "hog": "HOG stands for Histogram of Oriented Gradients, used for object detection by analyzing gradient orientations.",
        # Add more descriptions as needed...
        "gabor": "Gabor filters are used to detect specific frequencies and orientations, excellent for texture analysis.",
    "laplace": "Laplace filter highlights regions of rapid intensity change for edge detection.",
    "sobel": "Sobel filter calculates the gradient to highlight edges in horizontal and vertical directions.",
    "prewitt": "Prewitt operator is another method to approximate the image gradient for edge detection.",
    "roberts": "Roberts cross is an operator for edge detection using diagonal differences.",
    "bilateral": "Bilateral filter reduces noise but preserves edges via intensity and spatial filtering.",
    "gaussian blur": "Gaussian blur uses a Gaussian kernel to reduce image noise and detail.",
    "median": "Median filter replaces each pixel with the median of its neighborhood, removing salt-and-pepper noise.",
    "dilation": "Morphological dilation expands white regions in binary images.",
    "erosion": "Morphological erosion shrinks white regions in binary images.",
    "opening": "Opening is erosion followed by dilation, removing small objects while preserving shape.",
    "closing": "Closing is dilation followed by erosion, filling small holes while preserving shape.",
    "canny": "Canny edge detector uses gradient filtering, non-max suppression, and thresholds to find edges.",
    "harris": "Harris corner measures local auto-correlation to detect corners in an image.",
    "mser": "MSER finds stable regions across intensity thresholds, used for robust blob detection.",
    "adaptive threshold": "Adaptive thresholding calculates local thresholds for each region of an image.",
    "otsu": "Otsu thresholding computes an optimal threshold by maximizing inter-class variance.",
    "fourier": "Fourier Transform converts the image into frequency domain, useful for filtering.",
    "hist equalization": "Histogram equalization redistributes intensity to improve contrast.",
    "clahe": "CLAHE applies limited local histogram equalization to reduce noise amplification.",
    "hough lines": "Hough transform for lines maps points into parameter space to detect straight lines.",
    "watershed": "Watershed treats the image as a topographic surface, filling basins from minimums.",
    "scharr": "Scharr is an edge detection operator with improved rotation invariance over Sobel.",
    "blob": "LoG-based blob detection finds regions that differ in brightness or texture.",
    "median denoising": "Median filtering removes impulsive noise by replacing each pixel with the median of neighbors.",
    "log": "Laplacian of Gaussian emphasizes edges by combining Gaussian smoothing and Laplacian.",
    "connected components": "Labels distinct groups of connected pixels in a binary image.",
    "skeletonization": "Skeletonization reduces shapes to their minimal form while preserving connectivity.",
    "k-means": "K-means clustering partitions image pixels into k clusters by intensity.",
    "mean shift": "Mean Shift moves a window to find density modes for segmentation.",
    "dog": "Difference of Gaussians approximates Laplacian of Gaussian for edge and detail detection.",
    "ridge": "Ridge filter emphasizes line-like structures or ridges in the image.",
    "contours": "Contour detection finds continuous curves along boundaries of objects."
    }
    
    question = question.lower()
    if "guide" in question or "tour" in question:
        return "Welcome! Ask about any feature like 'SIFT', 'HOG', or 'What is watershed?'"
    for key, desc in feature_descriptions.items():
        if key in question:
            return desc
    return "I don't recognize that feature. Type 'guide' for a hint or ask about SIFT, HOG, Gabor, etc."

def chatbot_component():
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    user_input = st.text_input("Ask about any feature:")
    if st.button("Ask"):
        st.session_state["chat_history"].append(("User", user_input))
        response = chatbot_reply(user_input)
        st.session_state["chat_history"].append(("Bot", response))

    for speaker, text in st.session_state["chat_history"]:
        st.markdown(f"**{speaker}:** {text}")

###################
# Streamlit main (with Independent Components)
###################
def main():
    st.title("AI4Mars Unified Analysis")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Feature Analysis", "Landing Safety Analysis", "Chatbot"])

    if selection == "Feature Analysis":
        st.header("Feature + Filter Analysis")
        uploaded_file = st.file_uploader("Upload an Image for Feature Analysis", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 0)  # Grayscale
            feature_analysis_component(img)

    elif selection == "Landing Safety Analysis":
        st.header("Landing Safety Analysis")
        data_dir = st.text_input("Enter dataset directory:", "ai4mars-dataset-merged-0.1")
        ground_truth_input = st.text_area("Enter ground truth labels (comma-separated):", "safe,unsafe,safe")
        ground_truth_labels = [label.strip() for label in ground_truth_input.split(",")]
        num_images = st.number_input("Number of images to process:", value=5, min_value=1)

        if st.button("Run Safety Analysis"):
            landing_safety_component(data_dir, ground_truth_labels, num_images)

    elif selection == "Chatbot":
        st.header("Image Processing Feature Chatbot")
        chatbot_component()

if __name__ == "__main__":
    main()
