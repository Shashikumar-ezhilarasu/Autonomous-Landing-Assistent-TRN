import os
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

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

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        preprocessed_image = preprocess_image(gray)

        safety_status = assess_landing_safety(preprocessed_image)
        if safety_status == "safe":
            safe_count += 1
        else:
            unsafe_count += 1

        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        axs[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axs[0, 0].set_title(f"Safety Status: {safety_status}")
        axs[0, 0].axis('off')

        axs[1, 0].imshow(preprocessed_image, cmap='gray')
        axs[1, 0].set_title(f"Processed Image")
        axs[1, 0].axis('off')

        st.pyplot(fig)
        plt.close(fig)

        if i < len(ground_truth_labels):
            st.write(f"Ground Truth: {ground_truth_labels[i]}")

    if total_images > 0:
        accuracy = (correct_predictions / total_images) * 100
        st.write(f"Model Accuracy: {accuracy:.2f}%")

        if safe_count > unsafe_count:
            st.subheader("Overall Conclusion: Safe to Land")
        else:
            st.subheader("Overall Conclusion: Not Safe")
