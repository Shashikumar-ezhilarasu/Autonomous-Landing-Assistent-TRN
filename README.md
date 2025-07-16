 
# Terrain Relative Navigation (TRN) - An Spacecraft assistent 
<img width="1470" height="759" alt="image" src="https://github.com/user-attachments/assets/ba6fb8bc-427c-4786-a422-33562fc75d71" />
<img width="1470" height="759" alt="image" src="https://github.com/user-attachments/assets/240f4a17-d2ea-47a7-a48a-b782daf16e6d" />

 

This repository implements a comprehensive suite of image processing tools for Terrain Relative Navigation (TRN) systems. These features can be utilized for pre-processing, feature extraction, noise handling, and segmentation in applications such as autonomous navigation, remote sensing, and terrain analysis.
#PROBLEM STATEMENT

 Develop a TRN system using image data, Lidar, and IMU to ensure precise and safe autonomous landings.Unique Feature: Combines real-time terrain mapping with machine learning for autonomous decision-making.Main Problem Addressed: Reduces mission failure risk by accurately identifying and landing in safe zones on
diverse planetary surfaces.

# PROPOSED SOLUTION
Develop an AI/ML-driven Terrain Relative Navigation(TRN) system that leverages image data, Lidar, and IMU to enable precise and autonomous spacecraft landings.The solution should ensure real-time terrain mapping and safe zone identification, reducing mission failure risks on diverse planetary surfaces
## Features

### Edge Detection and Feature Extraction
1. Canny Edge Detection  
2. Sobel Filter  
3. Prewitt Filter  
4. Scharr Filter  
5. Laplacian of Gaussian (LoG)  
6. Harris Corner Detection  

### Morphological Operations
7. Dilation  
8. Erosion  
9. Opening  
10. Closing  

### Noise Reduction
11. Bilateral Filter Denoising  
12. Median Filtering (Scipy)  

### Histogram and Contrast Adjustments
13. Histogram Equalization  
14. CLAHE (Contrast Limited Adaptive Histogram Equalization)  

### Segmentation and Clustering
15. K-means Segmentation  
16. Watershed Segmentation  

### Transformations
17. Fourier Transform for Frequency Analysis  
18. Gaussian Smoothing  

### Skeletonization and Structure Analysis
19. Skeletonization  
20. Gaussian Laplace for Blob Detection  

### Keypoint Detection
21. Corner Detection with Harris  
22. Segmentation Markers  

### Filters for Terrain Analysis
23. Thresholding (e.g., Otsu's)  
24. Edge and Ridge Detection (e.g., Sobel/Scharr)  

### Image Noise Augmentation (Synthetic Testing)
25. Gaussian Noise Addition  

### Topography-Specific Features
26. Gradient-Based Feature Detection (Prewitt/Sobel)  
27. Multi-resolution Analysis (Wavelets or Fourier)  

### Spatial Transformations
28. Grayscale Conversion (for pre-processing)  
29. Histogram-based Texture Analysis  

### Shape and Contour Features
30. Contour Detection  
31. Watershed for Surface Segmentation  

### Image Analysis
32. Intensity-Based Thresholding  
33. Adaptive Thresholding  

### Data Augmentation
34. Random Noise Addition for Robustness  

### Structural Features
35. Texture Segmentation  
36. Blurring for Focused Edge Detection  

### Color Space Analysis
37. Conversion to Grayscale  
38. RGB Analysis for Terrain Classification  

### Cluster Analysis
39. Pixel Clustering using K-means  
40. Feature Clustering for TRN Models  

## Installation

1. Clone this repository:
   bash
   git clone https://github.com/your-username/TRN-Image-Processing-Toolbox.git
   
2. Navigate to the repository directory:
   bash
   cd TRN-Image-Processing-Toolbox
   
3. Install required dependencies:
   bash
   pip install -r requirements.txt
   

## Usage

Import the desired modules and functions in your Python scripts. Example usage:

python
from trn_toolbox import edge_detection

# Apply Canny Edge Detection
edges = edge_detection.canny(input_image)

pip install streamlit opencv-contrib-python numpy matplotlib scikit-image scipy

For more examples and usage instructions, check the examples/ folder.

## Applications

This toolbox is particularly suited for:
- Autonomous vehicle navigation
- Mars rover and planetary exploration systems
- Satellite image processing
- Environmental monitoring and terrain mapping




