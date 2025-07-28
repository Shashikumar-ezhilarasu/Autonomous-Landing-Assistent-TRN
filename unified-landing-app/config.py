"""
Configuration settings for the Autonomous Landing Assessment System
"""

# Safety Assessment Thresholds
SAFETY_THRESHOLDS = {
    'max_slope': 15.0,           # Maximum allowable slope in degrees
    'max_roughness': 1000.0,     # Maximum surface roughness
    'max_edge_density': 0.1,     # Maximum edge density ratio
    'max_texture_variance': 500.0 # Maximum texture variance
}

# Scoring Weights (must sum to 1.0)
SCORING_WEIGHTS = {
    'slope': 0.3,        # 30% weight for slope analysis
    'roughness': 0.3,    # 30% weight for roughness analysis
    'edge_density': 0.2, # 20% weight for edge analysis
    'texture': 0.2       # 20% weight for texture analysis
}

# Safety Classification Ranges
SAFETY_RANGES = {
    'excellent': {'min': 80, 'color': '#4CAF50', 'status': 'EXCELLENT'},
    'good': {'min': 60, 'color': '#FFC107', 'status': 'GOOD'},
    'moderate': {'min': 40, 'color': '#FF9800', 'status': 'MODERATE'},
    'poor': {'min': 0, 'color': '#f44336', 'status': 'POOR'}
}

# Computer Vision Parameters
CV_PARAMS = {
    'canny_low_threshold': 100,
    'canny_high_threshold': 200,
    'sobel_kernel_size': 3,
    'hough_lines_threshold': 80,
    'sift_num_features': 0,  # 0 = unlimited
    'gaussian_blur_kernel': (5, 5),
    'median_blur_kernel': 5
}

# UI Configuration
UI_CONFIG = {
    'page_title': '🚀 Autonomous Landing Site Assessment',
    'page_icon': '🛸',
    'layout': 'wide',
    'primary_color': '#FF6B35',
    'background_color': '#f0f2f6',
    'text_color': '#262730'
}

# File Upload Settings
UPLOAD_CONFIG = {
    'max_file_size': 50,  # MB
    'allowed_extensions': ['png', 'jpg', 'jpeg'],
    'image_resize_max': 1024  # Maximum dimension for processing
}
