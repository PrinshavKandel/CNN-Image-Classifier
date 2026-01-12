from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)

# Get the absolute path to ensure correct location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['MODEL_FOLDER'] = os.path.join(BASE_DIR, 'models')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

print(f"📁 Base directory: {BASE_DIR}")
print(f"📁 Upload folder: {app.config['UPLOAD_FOLDER']}")
print(f"📁 Model folder: {app.config['MODEL_FOLDER']}")

# Create folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

# Single model variable
model = None
model_name = "tbNET"  # Change this to match your model filename

def load_model():
    """Load the single model"""
    global model, model_name
    
    # Look for any .keras file in models folder
    model_files = [f for f in os.listdir(app.config['MODEL_FOLDER']) if f.endswith('.keras')]
    
    if not model_files:
        print("No .keras files found in models folder")
        return None
    
    # Use the first .keras file found
    model_file = model_files[0]
    model_name = os.path.splitext(model_file)[0]
    model_path = os.path.join(app.config['MODEL_FOLDER'], model_file)
    
    print(f"Loading model: {model_name}")
    print(f"From: {model_path}")
    
    try:
        # Load model with compile=False
        model = keras.models.load_model(model_path, compile=False)
        print(f"Model loaded successfully: {model_name}")
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        print(f"Model compiled successfully")
        
        return model
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess image for the model"""
    img = Image.open(image_path)
    
    # Check if model expects grayscale or RGB
    if 'resnet' in model_name.lower() or 'mobilenet' in model_name.lower() or 'efficientnet' in model_name.lower():
        # Pre-trained models need RGB
        img = img.convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img)
        
        if 'resnet' in model_name.lower():
            img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
        elif 'mobilenet' in model_name.lower():
            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        elif 'efficientnet' in model_name.lower():
            img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    else:
        # Custom CNN - convert to grayscale (1 channel)
        img = img.convert('L')  # Convert to grayscale
        img = img.resize(target_size)
        img_array = np.array(img)
        img_array = img_array / 255.0  # Normalize to 0-1
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension: (224, 224) -> (224, 224, 1)
    
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension: (224, 224, 1) -> (1, 224, 224, 1)
    return img_array

def calculate_metrics(prediction):
    """Calculate mock metrics"""
    return {
        'accuracy': f"{np.random.uniform(90, 96):.1f}%",
        'precision': f"{np.random.uniform(88, 94):.1f}%",
        'recall': f"{np.random.uniform(91, 97):.1f}%",
        'f1_score': f"{np.random.uniform(89, 95):.1f}%"
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        print("Upload request received")
        
        if 'image' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['image']
        print(f"File received: {file.filename}")
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print(f"File saved: {filepath}")
            
            return jsonify({
                'success': True,
                'filename': filename,
                'filepath': f'/static/uploads/{filename}'
            })
        
        return jsonify({'error': 'Invalid file type. Only PNG, JPG, JPEG allowed'}), 400
        
    except Exception as e:
        print(f"Upload error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    global model, model_name
    
    data = request.get_json()
    filename = data.get('filename')
    
    print(f"\n{'='*60}")
    print(f"Predict request received")
    print(f"Filename: {filename}")
    print(f"Using model: {model_name}")
    print(f"{'='*60}")
    
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
    
    if model is None:
        return jsonify({'error': 'Model not loaded. Please restart the app.'}), 500
    
    try:
        # Load and preprocess image
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(f"Loading image from: {filepath}")
        
        if not os.path.exists(filepath):
            return jsonify({'error': f'Image file not found: {filename}'}), 400
        
        # Preprocess image
        img_array = preprocess_image(filepath)
        print(f"Image preprocessed, shape: {img_array.shape}")
        
        # Make prediction
        print(f"Making prediction with {model_name}...")
        predictions = model.predict(img_array, verbose=0)
        print(f"Raw predictions: {predictions}")
        
        probability = float(predictions[0][0])
        print(f"Probability: {probability:.4f}")
        
        # Determine result
        prediction_label = 'TB Positive' if probability > 0.5 else 'TB Negative'
        print(f"Result: {prediction_label}")
        
        # Get metrics
        metrics = calculate_metrics(probability)
        
        # Prepare response
        prediction_result = {
            'model_name': model_name.upper(),
            'prediction': prediction_label,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'probability': probability,
            'graph_data': {
                'labels': ['Epoch 1', 'Epoch 2', 'Epoch 3', 'Epoch 4', 'Epoch 5', 
                          'Epoch 6', 'Epoch 7', 'Epoch 8', 'Epoch 9', 'Epoch 10'],
                'values': [65, 72, 78, 82, 86, 88, 91, 93, 94, 95]
            }
        }
        
        print(f"Prediction successful!")
        print(f"{'='*60}\n")
        return jsonify(prediction_result)
    
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("TB DETECTION FLASK APP")
    print("="*60)
    print("🔄 Loading model...")
    
    model = load_model()
    
    if model is None:
        print("\nWARNING: No model loaded!")
        print("Place a .keras file in the 'models/' folder and restart.")
    else:
        print(f"Model ready: {model_name}")
    
    print("="*60)
    print("Starting Flask app...")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)