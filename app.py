from flask import Flask, render_template, jsonify, request
import os
import hashlib
from utils.predict import predict_single_image
from utils.history import add_prediction_history, get_recent_predictions

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/input_images'
# Allowed image formats
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ============================================================================
# UTILITY FUNCTION: Check if file has an allowed extension
# ============================================================================
def allowed_file(filename):
    """Check if uploaded file has an allowed image extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# ============================================================================
# ROUTE: Home Page
# ============================================================================
@app.route('/')
def index():
    """Render the home page"""
    return render_template('index.html', recent_predictions=get_recent_predictions(limit=10))

# ============================================================================
# ROUTE: Upload single fingerprint image
# ============================================================================
@app.route('/upload-image', methods=['POST'])
def upload_image():
    """
    Handle single image upload
    - Validates that exactly 1 file is uploaded
    - Validates file format
    - Clears previous images and saves new one
    """
    try:
        # Check if file is present in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded. Please select an image.'})

        file = request.files['file']

        # Check if file was selected
        if file.filename == '':
            return jsonify({'error': 'No file selected. Please choose an image.'})

        # Validate file extension
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file format. Please upload PNG, JPG, JPEG, BMP, or GIF.'})

        # Clear old images from the upload directory
        for f in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], f)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Warning: Could not delete {f}: {e}")

        # Save the uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        print(f"✅ Image uploaded successfully: {file.filename}")
        return jsonify({'success': True, 'filename': file.filename})

    except Exception as e:
        print(f"❌ Upload Error: {str(e)}")
        return jsonify({'error': f'Upload failed: {str(e)}'})

# ============================================================================
# ROUTE: Predict blood group from uploaded image
# ============================================================================
@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict blood group from the uploaded fingerprint image
    - Validates that an image exists
    - Preprocesses the image (grayscale, resize, normalize)
    - Passes to CNN model for prediction
    - Returns blood group and confidence score
    """
    try:
        input_dir = app.config['UPLOAD_FOLDER']
        files = os.listdir(input_dir)

        # Check if an image has been uploaded
        if len(files) == 0:
            return jsonify({'error': 'No image uploaded. Please upload a fingerprint image first.'})

        # Get the first (and only) image file
        image_filename = files[0]
        image_path = os.path.join(input_dir, image_filename)

        print(f"📸 Processing image: {image_filename}")

        # Predict blood group for the single image
        result = predict_single_image(image_path)

        # Lightweight runtime history tracking
        with open(image_path, 'rb') as image_file:
            image_hash = hashlib.sha256(image_file.read()).hexdigest()[:12]
        image_id = f"{image_filename} ({image_hash})"
        add_prediction_history(
            image_id=image_id,
            blood_group=result['label'],
            confidence_score=result['confidence_score']
        )

        return jsonify({
            'success': True,
            'filename': image_filename,
            'blood_group': result['label'],
            'confidence': result['confidence_score'],
            'all_probabilities': result['confidence'],
            'recent_predictions': get_recent_predictions(limit=10)
        })

    except ValueError as e:
        # Handle image reading errors
        print(f"❌ Invalid Image: {str(e)}")
        return jsonify({'error': f'Invalid image: {str(e)}'})
    except Exception as e:
        # Handle other errors
        print(f"❌ Prediction Error: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'})

# ============================================================================
# MAIN: Start Flask server
# ============================================================================
if __name__ == '__main__':
    app.run(debug=True)