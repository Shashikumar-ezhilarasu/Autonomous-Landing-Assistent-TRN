pip install streamlit opencv-contrib-python numpy matplotlib scikit-image scipy


# Terrain Relative Navigation (TRN) Image Processing Toolbox

This repository implements a comprehensive suite of image processing tools for Terrain Relative Navigation (TRN) systems. These features can be utilized for pre-processing, feature extraction, noise handling, and segmentation in applications such as autonomous navigation, remote sensing, and terrain analysis.

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


For more examples and usage instructions, check the examples/ folder.

## Applications

This toolbox is particularly suited for:
- Autonomous vehicle navigation
- Mars rover and planetary exploration systems
- Satellite image processing
- Environmental monitoring and terrain mapping

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch:
   bash
   git checkout -b feature/your-feature-name
   
3. Commit your changes:
   bash
   git commit -m "Add your message here"
   
4. Push the branch:
   bash
   git push origin feature/your-feature-name
   
5. Open a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- Developed with support from the open-source Python image processing ecosystem.
- Special thanks to contributors and maintainers of libraries like OpenCV, SciPy, and scikit-image.

