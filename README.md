pip install streamlit opencv-contrib-python numpy matplotlib scikit-image scipy


Image Processing with Feature and Landing Safety Analysis
This project is a Streamlit-based web application that performs various image processing techniques for feature extraction and landing safety assessment. The app allows users to upload an image, which is then processed to compute key features (like SIFT, HOG, Gabor filters, and others) and evaluate the surface slope and roughness for assessing landing safety.

Features
Feature Analysis
Keypoint Detection and Feature Extraction: Using techniques such as SIFT, HOG, Gabor filters, Sobel, Prewitt, and more.
Image Filtering: Apply various filters such as Gaussian blur, median filtering, and edge detection (e.g., Canny, Sobel).
Morphological Operations: Operations like dilation, erosion, opening, closing, and watershed segmentation to analyze image structure.
Noise and Thresholding: Add noise, apply adaptive thresholding, and apply Otsu’s method for binarization.
Blob Detection: Using LoG for blob detection and connected components for region detection.
Fourier Transform: Apply FFT for frequency-domain analysis.
Landing Safety Analysis
Surface Slope Calculation: Using Sobel operator to determine the gradient and assess slope.
Surface Roughness Calculation: Using Laplacian operator to measure surface roughness.
Safety Assessment: Based on predefined thresholds, classify the landing surface as "safe" or "unsafe."
Installation
Prerequisites
To run the application, you need Python 3.6 or higher. Install the required libraries using the following:

bash
Copy code
pip install streamlit opencv-python scikit-image matplotlib pillow scipy
Running the Application
To run the app, save the Python script (e.g., app.py) and use the following command in the terminal:

bash
Copy code
streamlit run app.py
Once the app starts, it will be available at http://localhost:8501 in your browser.

Usage
Upload Image
Feature Analysis: After uploading an image, the app will automatically process and display various feature extractions and filters applied to the image.
Landing Safety Analysis: The app will calculate the surface slope and roughness, and evaluate whether the landing area is "safe" or "unsafe" based on the predefined thresholds.
Available Image Processing Features:
SIFT: Detects keypoints and descriptors in the image.
HOG: Computes Histogram of Oriented Gradients for object detection.
Gabor Filter: Applies a Gabor filter for texture analysis.
Gaussian Laplace: Used for edge detection.
Sobel, Prewitt, Roberts: Apply edge detection filters.
Canny Edge Detection: Detects edges in the image.
Morphological Operations: Apply dilation, erosion, opening, and closing operations.
Surface Analysis: Calculate the surface slope and roughness for landing safety evaluation.
Fourier Transform: Applies the Fourier transform to the image for frequency-domain analysis.
Landing Safety Evaluation:
The app will display if the landing area is "safe" or "unsafe" based on slope and roughness calculations.
Example
Once you upload an image, you will see processed results with the following outputs:

Feature Extractions: Keypoints, filtered images, and edge detection results.
Surface Analysis: Slope and roughness calculations with a classification for landing safety.
Code Explanation
process_all_features(img):
This function processes an input image using various feature extraction techniques, returning a list of processed results.

feature_analysis_component(img):
This function displays the results of the feature extraction process on the Streamlit app in a grid layout.

calculate_surface_slope(image) and calculate_surface_roughness(image):
These functions calculate the slope and roughness of the image, essential for determining landing safety.

assess_landing_safety(image):
Based on the slope and roughness values, this function classifies the landing area as either "safe" or "unsafe."

Contributing
Feel free to fork this repository, contribute improvements or report any issues.

