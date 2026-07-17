"""
Fingerprint Blood Group Prediction Module
==========================================
This module handles preprocessing and prediction of blood groups
from a single fingerprint image using a trained EfficientNetB0 CNN model.

Functions:
  - preprocess_image(): Convert raw fingerprint image to model input format
  - predict_single_image(): Predict blood group and get confidence scores
"""

import os
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path

# MODEL LOADING
# Load the trained CNN model (EfficientNetB0)
MODEL_PATH = Path(__file__).parent.parent / "model" / "final_best_efficientnetb0_model_final.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Blood group class labels
CLASS_LABELS = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']

# Target image size (must match training data)
TARGET_SIZE = (103, 96)

# ============================================================================
# PREPROCESSING FUNCTION
# ============================================================================
def enhance_fingerprint(image, apply_sharpening=True):
    """
    Enhance a fingerprint image for cleaner inference input.

    Steps:
      1. Ensure grayscale image
      2. Reduce noise with Gaussian blur
      3. Improve local contrast with CLAHE
      4. Apply mild sharpening for ridge visibility

    Args:
        image (np.ndarray): Input image (grayscale or BGR)
        apply_sharpening (bool): Whether to apply mild sharpening

    Returns:
        np.ndarray: Enhanced grayscale image
    """
    if image is None:
        raise ValueError("Invalid image provided for enhancement.")

    # 1) Convert to grayscale if image is in BGR/RGB format
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 2) Light noise reduction
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)

    # 3) Local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(denoised)

    if apply_sharpening:
        # 4) Optional sharpening (kept mild for stability)
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(contrast_enhanced, -1, sharpen_kernel)

    return contrast_enhanced


def preprocess_image(img_path, use_enhancement=True):
    """
    Preprocess fingerprint image for model prediction.
    
    Steps:
      1. Load image in grayscale mode
      2. Resize to TARGET_SIZE (103, 96)
      3. Normalize pixel values to [0, 1] range
      4. Expand dimensions to match model input shape (1, 103, 96, 1)
      5. Convert grayscale to RGB (expand to 3 channels) for model compatibility
    
    Args:
        img_path (str): Path to the fingerprint image file
    
    Returns:
        np.ndarray: Preprocessed image with shape (1, 103, 96, 3) ready for model
    
    Raises:
        ValueError: If image cannot be read or is invalid
    """
    
    # Load image in grayscale mode for stable baseline behavior
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}. Make sure it's a valid image file.")
    
    # Optional fingerprint enhancement pipeline before resizing
    if use_enhancement:
        img = enhance_fingerprint(img)

    # Resize image to TARGET_SIZE (which the model expects)
    img = cv2.resize(img, TARGET_SIZE)
    
    # Normalize pixel values from [0, 255] to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Add channel dimension: (103, 96) -> (103, 96, 1)
    img = np.expand_dims(img, axis=-1)
    
    # Convert grayscale to RGB by repeating the channel 3 times
    # (103, 96, 1) -> (103, 96, 3)
    img = np.repeat(img, 3, axis=-1)
    
    # Add batch dimension: (103, 96, 3) -> (1, 103, 96, 3)
    img = np.expand_dims(img, axis=0)
    
    return img

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================
def predict_single_image(img_path, use_enhancement=True):
    """
    Predict blood group from a single fingerprint image.
    
    Process:
      1. Preprocess the image
      2. Feed to trained CNN model
      3. Get probability predictions for all 8 classes
      4. Find the class with highest probability
      5. Calculate confidence score (max probability × 100)
    
    Args:
        img_path (str): Path to the fingerprint image file
    
    Returns:
        dict: Dictionary containing:
          - 'filename': Name of the image file
          - 'label': Predicted blood group (e.g., 'O+', 'A-', etc.)
          - 'confidence_score': Maximum confidence as percentage (0-100)
          - 'confidence': List of probabilities for all 8 classes
    
    Example:
        >>> result = predict_single_image('fingerprint.jpg')
        >>> print(result)
        {
            'filename': 'fingerprint.jpg',
            'label': 'O+',
            'confidence_score': 92.45,
            'confidence': [0.02, 0.01, 0.03, 0.0, 0.01, 0.01, 0.9245, 0.001]
        }
    """
    
    try:
        print(f"\n{'='*60}")
        print(f"📸 Processing image: {os.path.basename(img_path)}")
        print(f"{'='*60}")
        
        # Step 1: Preprocess the image
        print("🧹 Step 1: Preprocessing image...")
        print(f"   - Loading in grayscale mode")
        print(f"   - Resizing to {TARGET_SIZE}")
        print(f"   - Normalizing pixel values [0, 1]")
        img_tensor = preprocess_image(img_path, use_enhancement=use_enhancement)
        print(f"   - Final shape: {img_tensor.shape}")
        
        # Step 2: Get model prediction
        print("🧠 Step 2: Running model prediction...")
        predictions = model.predict(img_tensor, verbose=0)[0]
        
        # Step 3: Find predicted class
        predicted_index = int(np.argmax(predictions))
        predicted_label = CLASS_LABELS[predicted_index]
        
        # Step 4: Calculate confidence score
        confidence_percentage = float(predictions[predicted_index] * 100)
        
        # Log results
        print(f"✅ Prediction complete!")
        print(f"   - Blood Group: {predicted_label}")
        print(f"   - Confidence: {confidence_percentage:.2f}%")
        print(f"\n📊 All class probabilities:")
        for i, (label, prob) in enumerate(zip(CLASS_LABELS, predictions)):
            bar_length = int(prob * 30)
            bar = '█' * bar_length + '░' * (30 - bar_length)
            print(f"   {label:4s} {bar} {prob*100:6.2f}%")
        print(f"{'='*60}\n")
        
        # Return results
        return {
            'filename': os.path.basename(img_path),
            'label': predicted_label,
            'confidence_score': confidence_percentage,
            'confidence': predictions.tolist()
        }
    
    except ValueError as e:
        print(f"❌ Preprocessing error: {e}")
        raise
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        raise

# ============================================================================
# OLD FUNCTIONS (KEPT FOR REFERENCE - NOT USED)
# ============================================================================
# These functions were used for the 10-image approach and are kept
# for backwards compatibility if needed in the future.

def predict_all(image_dir):
    """
    DEPRECATED: Predict blood group for all images in a directory.
    Use predict_single_image() for single image prediction instead.
    """
    predictions = []
    print("📂 Files found:", os.listdir(image_dir))
    
    for filename in sorted(os.listdir(image_dir)):
        if filename.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png')):
            full_path = os.path.join(image_dir, filename)
            try:
                result = predict_single_image(full_path)
                predictions.append(result)
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
    
    print("✅ Predictions count:", len(predictions))
    return predictions


def majority_prediction(predictions):
    """
    DEPRECATED: Find majority blood group from multiple predictions.
    Not used in single-image prediction mode.
    """
    labels = [p['label'] for p in predictions]
    return max(set(labels), key=labels.count) if labels else None