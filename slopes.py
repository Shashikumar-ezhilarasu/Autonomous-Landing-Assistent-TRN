import numpy as np
import cv2
import streamlit as st
import matplotlib.pyplot as plt

# Function to calculate surface slope
def calculate_surface_slope(image):
    return np.random.uniform(0, 15)

# Function to calculate surface roughness
def calculate_surface_roughness(image):
    return np.random.uniform(0, 5)

# Function to assess landing safety
def assess_landing_safety(image):
    slope = calculate_surface_slope(image)
    roughness = calculate_surface_roughness(image)

    max_slope = 15
    max_roughness = 5

    status = ""
    if slope > max_slope:
        status += "Warning: Surface slope is too steep.\n"
    else:
        status += "Surface slope is safe.\n"

    if roughness > max_roughness:
        status += "Warning: Surface roughness is too high.\n"
    else:
        status += "Surface roughness is safe.\n"
    
    return status

# Function to process and display SIFT features
def process_uploaded_images(uploaded_files):
    sift = cv2.SIFT_create()

    for i, uploaded_file in enumerate(uploaded_files):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            st.write(f"Error reading file: {uploaded_file.name}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        image_with_keypoints = cv2.drawKeypoints(
            image, keypoints, None,
            flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
        )

        safety_status = assess_landing_safety(image)

        st.write(f"**Image {i+1}:** {uploaded_file.name}")
        fig, ax = plt.subplots()
        ax.imshow(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        st.pyplot(fig)

        st.write("Safety Status:")
        st.write(safety_status)

# Streamlit UI
st.title("SIFT Image Processing and Landing Safety Assessment")

uploaded_files = st.file_uploader(
    "Drag and drop or upload your images:",
    accept_multiple_files=True,
    type=["png", "jpg", "jpeg"]
)

if st.button("Process Images"):
    if uploaded_files:
        process_uploaded_images(uploaded_files)
    else:
        st.write("Please upload at least one image.")
