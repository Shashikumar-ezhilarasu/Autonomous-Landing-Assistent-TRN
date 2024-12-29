import os
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

# Helper functions for image processing and safety assessment

def preprocess_image(image):
    """Preprocess the image (e.g., noise reduction)."""
    return cv2.GaussianBlur(image, (5, 5), 0)

def calculate_surface_slope(image):
    """Calculate surface slope using Sobel gradients."""
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    gradient_magnitude = cv2.magnitude(grad_x, grad_y)
    slope = np.max(gradient_magnitude)  # Max gradient as slope
    return slope

def calculate_surface_roughness(image):
    """Calculate surface roughness using GLCM."""
    glcm = graycomatrix(image, [1], [0], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    return contrast

def extract_features(image):
    """Extract relevant features (e.g., slope, roughness) for model input."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    slope = calculate_surface_slope(gray)
    roughness = calculate_surface_roughness(gray)
    return [slope, roughness]

def train_model(features, labels):
    """Train a Random Forest model and evaluate using cross-validation."""
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')  # Handling class imbalance
    
    # Cross-validation to evaluate model performance
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, features, labels, cv=cv, scoring='accuracy')
    
    # Train the model on the full dataset
    model.fit(features, labels)
    
    return model, cv_scores

def assess_landing_safety_with_model(image, model):
    """Assess safety using a trained model."""
    features = extract_features(image)
    safety_status = model.predict([features])
    return "safe" if safety_status == 1 else "unsafe"

def load_and_display_images(data_dir, image_subfolder, model, max_images=100):
    """Load and display images with safety assessments."""
    full_path = os.path.join(data_dir, image_subfolder)
    if not os.path.exists(full_path):
        st.error(f"The directory {full_path} does not exist. Please check the path.")
        return [], []

    image_files = sorted([f for f in os.listdir(full_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))])
    image_files = image_files[:min(max_images, 100)]
    
    safe_count = 0
    unsafe_count = 0
    images_processed = []

    # Display progress bar
    progress_bar = st.progress(0)

    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(full_path, image_file)
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        # Assess landing safety using the trained model
        safety_status = assess_landing_safety_with_model(image, model)
        
        if safety_status == "safe":
            safe_count += 1
        else:
            unsafe_count += 1
        
        # Display the results
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original image
        axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axs[0].set_title(f"Original Image: {image_file}")
        axs[0].axis('off')
        
        # Preprocessed (grayscale) image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        preprocessed_image = preprocess_image(gray)
        axs[1].imshow(preprocessed_image, cmap='gray')
        axs[1].set_title(f"Safety Status: {safety_status}")
        axs[1].axis('off')
        
        st.pyplot(fig)
        plt.close(fig)

        # Track processed images
        images_processed.append(image_file)
        
        # Update the progress bar
        progress = (idx + 1) / len(image_files)
        progress_bar.progress(progress)

    return safe_count, unsafe_count, images_processed

def display_metrics(cv_scores, accuracy, precision, recall, f1, cm, images_processed):
    """Display cross-validation and evaluation metrics in Streamlit."""
    st.markdown(f"**Model Cross-Validation Accuracy (5-fold):** {np.mean(cv_scores) * 100:.2f}%")
    st.markdown(f"**Model Accuracy on Test Set:** {accuracy * 100:.2f}%")

    # Handle edge case where no positive class exists in predictions
    if precision == 0 and recall == 0:
        st.markdown(f"**Precision:** Not Defined (no positive predictions)")
        st.markdown(f"**Recall:** Not Defined (no positive predictions)")
        st.markdown(f"**F1 Score:** Not Defined (no positive predictions)")
    else:
        st.markdown(f"**Precision:** {precision * 100:.2f}%")
        st.markdown(f"**Recall:** {recall * 100:.2f}%")
        st.markdown(f"**F1 Score:** {f1 * 100:.2f}%")
    
    st.write("Confusion Matrix:")
    st.write(cm)
    
    # Display processed images count
    st.write(f"Processed {len(images_processed)} images")

def main():
    st.title("Landing Safety Assessment Tool")
    st.write("Upload a dataset directory and assess landing safety.")

    # Input for dataset directory
    data_dir = st.text_input("Enter the path to the dataset directory:", value="ai4mars-dataset-merged-0.1")
    image_subfolder = st.text_input("Enter the relative path to the image subfolder:", value="msl/images/edr")
    
    # User input for number of images to process (max 100)
    num_images = st.number_input("Enter the number of images to process (max 100):", min_value=1, max_value=500, value=100)

    # User input for training the model
    if st.button("Train Model"):
        # Load and prepare the dataset for training
        all_features = []
        all_labels = []
        image_files = sorted([f for f in os.listdir(os.path.join(data_dir, image_subfolder)) if f.lower().endswith(('png', 'jpg', 'jpeg'))])

        # Collect features and labels from all images (assumes you have a label for each image)
        for image_file in image_files[:min(num_images, 100)]:
            image_path = os.path.join(data_dir, image_subfolder, image_file)
            image = cv2.imread(image_path)
            if image is None:
                continue

            features = extract_features(image)
            label = 1 if "safe" in image_file else 0  # Assume 'safe' in filename for label
            all_features.append(features)
            all_labels.append(label)

        # Train the model
        model, cv_scores = train_model(all_features, all_labels)

        # Predict on a held-out test set (80-20 split)
        X_train, X_test, y_train, y_test = train_test_split(all_features, all_labels, test_size=0.2, random_state=42)
        y_pred = model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=1)  # Add zero_division to handle edge cases
        recall = recall_score(y_test, y_pred, zero_division=1)
        f1 = f1_score(y_test, y_pred, zero_division=1)
        cm = confusion_matrix(y_test, y_pred)

        st.success("Model trained successfully!")
        
        # Display the metrics
        display_metrics(cv_scores, accuracy, precision, recall, f1, cm, [])

        # Display safety assessment results using the trained model
        safe_count, unsafe_count, images_processed = load_and_display_images(data_dir, image_subfolder, model, num_images)
        
        # Display final safety conclusion
        if safe_count > unsafe_count:
            st.subheader("Overall Conclusion: Safe to Land")
        else:
            st.subheader("Overall Conclusion: Not Safe")

if __name__ == "__main__":
    main()
