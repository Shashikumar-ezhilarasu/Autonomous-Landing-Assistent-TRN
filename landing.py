import os
import cv2
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Define helper functions
def preprocess_image(image):
    """Preprocess the image (dummy function, replace with your own logic)."""
    return cv2.GaussianBlur(image, (5, 5), 0)

@st.cache_data
def assess_landing_safety(image):
    """Assess the landing safety based on image properties."""
    slope = calculate_surface_slope(image)
    roughness = calculate_surface_roughness(image)
    max_slope = 15
    max_roughness = 5
    return "safe" if slope <= max_slope and roughness <= max_roughness else "unsafe"

def calculate_surface_slope(image):
    """Dummy slope calculation function (replace with real implementation)."""
    return np.random.uniform(0, 20)

def calculate_surface_roughness(image):
    """Dummy roughness calculation function (replace with real implementation)."""
    return np.random.uniform(0, 10)

def load_and_display_images(data_dir, image_subfolder):
    """Load and display images with safety assessments."""
    full_path = os.path.join(data_dir, image_subfolder)

    # Check if the directory exists
    if not os.path.exists(full_path):
        st.error(f"The directory {full_path} does not exist. Please check the path.")
        return

    # Get image files
    image_files = sorted([
        f for f in os.listdir(full_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    if not image_files:
        st.error(f"No image files found in the directory {full_path}. Ensure the directory contains valid image files.")
        return

    safe_count = 0
    unsafe_count = 0

    # Process each image
    for image_file in image_files:
        image_path = os.path.join(full_path, image_file)
        image = cv2.imread(image_path)
        if image is None:
            st.write(f"Error reading file: {image_path}. Skipping.")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        preprocessed_image = preprocess_image(gray)

        safety_status = assess_landing_safety(preprocessed_image)
        if safety_status == "safe":
            safe_count += 1
        else:
            unsafe_count += 1

        # Plot the images
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axs[0].set_title(f"Original Image: {image_file}")
        axs[0].axis('off')

        axs[1].imshow(preprocessed_image, cmap='gray')
        axs[1].set_title(f"Safety Status: {safety_status}")
        axs[1].axis('off')

        st.pyplot(fig)
        plt.close(fig)

    # Display overall conclusion
    total_images = safe_count + unsafe_count
    if total_images > 0:
        if safe_count > unsafe_count:
            st.subheader("Overall Conclusion: Safe to Land")
        else:
            st.subheader("Overall Conclusion: Not Safe")

# Streamlit app
def main():
    st.title("Landing Safety Assessment Tool")
    st.write("Upload a dataset directory and assess landing safety.")

    # Input for dataset directory
    data_dir = st.text_input("Enter the path to the dataset directory:", value="ai4mars-dataset-merged-0.1")
    image_subfolder = st.text_input("Enter the relative path to the image subfolder:", value="msl/images/edr")

    # Load and display images
    if st.button("Run Safety Assessment"):
        load_and_display_images(data_dir, image_subfolder)

if __name__ == "__main__":
    main()