"""
Handwritten Digit Recognition - Flask Backend
This Flask application serves as the backend API for the digit recognition system.
It loads the trained CNN model and provides an endpoint for predictions.
"""

from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import re
import os

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize Flask application
app = Flask(__name__)

# Global variable to hold the loaded model
model = None


def load_model():
    """
    Load the trained CNN model from the H5 file.
    This function is called once when the server starts.
    """
    global model
    try:
        print("Loading trained model...")
        model_path = os.path.join(BASE_DIR, 'digit_model.h5')
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure 'digit_model.h5' exists. Run train_model.py first.")


def preprocess_image(image_data):
    """
    Preprocess the image received from the frontend for prediction.
    Improved preprocessing to match MNIST format:
    1. Decode base64 image data
    2. Convert to grayscale
    3. Find bounding box of the digit and center it
    4. Resize to 20x20 and pad to 28x28 (like MNIST)
    5. Normalize pixel values
    """
    # Remove the data URL prefix (e.g., "data:image/png;base64,")
    image_data = re.sub('^data:image/.+;base64,', '', image_data)
    
    # Decode base64 string to bytes
    image_bytes = base64.b64decode(image_data)
    
    # Open image using PIL
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to grayscale (L mode = 8-bit grayscale)
    image = image.convert('L')
    
    # Convert to numpy array
    image_array = np.array(image)
    
    # The canvas has white (255) drawing on black (0) background
    # Find the bounding box of the digit (non-zero pixels)
    rows = np.any(image_array > 50, axis=1)
    cols = np.any(image_array > 50, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        # No digit drawn, return empty image
        return np.zeros((1, 28, 28, 1), dtype='float32')
    
    # Get bounding box coordinates
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # Extract the digit region
    digit = image_array[rmin:rmax+1, cmin:cmax+1]
    
    # Make the bounding box square by padding the shorter dimension
    height, width = digit.shape
    if height > width:
        # Pad width
        diff = height - width
        pad_left = diff // 2
        pad_right = diff - pad_left
        digit = np.pad(digit, ((0, 0), (pad_left, pad_right)), mode='constant', constant_values=0)
    elif width > height:
        # Pad height
        diff = width - height
        pad_top = diff // 2
        pad_bottom = diff - pad_top
        digit = np.pad(digit, ((pad_top, pad_bottom), (0, 0)), mode='constant', constant_values=0)
    
    # Resize the digit to 20x20 pixels (MNIST digits are 20x20 centered in 28x28)
    digit_image = Image.fromarray(digit.astype(np.uint8))
    digit_image = digit_image.resize((20, 20), Image.Resampling.LANCZOS)
    digit = np.array(digit_image)
    
    # Create a 28x28 image and center the 20x20 digit in it (4 pixel border)
    final_image = np.zeros((28, 28), dtype=np.uint8)
    final_image[4:24, 4:24] = digit
    
    # MNIST format: background = 0, digit = high values (up to 255)
    # Our canvas: background = black (0), drawing = white (255)
    # So our format already matches MNIST - NO inversion needed!
    
    # Normalize pixel values to range [0, 1]
    final_image = final_image.astype('float32') / 255.0
    
    # Reshape to match model input shape: (1, 28, 28, 1)
    final_image = final_image.reshape(1, 28, 28, 1)
    
    return final_image


@app.route('/')
def index():
    """
    Serve the main HTML page with the drawing canvas.
    """
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to predict the digit from the drawn image.
    Expects JSON with 'image' field containing base64 encoded image data.
    Returns JSON with predicted digit and confidence score.
    """
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please train the model first.'
            }), 500
        
        # Get image data from request
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'No image data provided'
            }), 400
        
        image_data = data['image']
        
        # Preprocess the image
        processed_image = preprocess_image(image_data)
        
        # Make prediction using the model
        # Returns array of probabilities for each digit (0-9)
        predictions = model.predict(processed_image, verbose=0)
        
        # Get the predicted digit (index with highest probability)
        predicted_digit = int(np.argmax(predictions[0]))
        
        # Get the confidence score (probability of the predicted digit)
        confidence = float(predictions[0][predicted_digit])
        
        # Get all probabilities for detailed response
        all_probabilities = {
            str(i): float(predictions[0][i]) 
            for i in range(10)
        }
        
        print(f"Predicted digit: {predicted_digit} with confidence: {confidence:.4f}")
        
        return jsonify({
            'success': True,
            'digit': predicted_digit,
            'confidence': confidence,
            'probabilities': all_probabilities
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/health')
def health():
    """
    Health check endpoint to verify the server and model are running.
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })


# Load the model when the application starts
load_model()


if __name__ == '__main__':
    # Run the Flask development server
    # debug=True enables auto-reload on code changes
    # host='0.0.0.0' makes it accessible from other machines
    print("\n" + "=" * 60)
    print("Handwritten Digit Recognition - Flask Server")
    print("=" * 60)
    print("Server starting on http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("=" * 60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
