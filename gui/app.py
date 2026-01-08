from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': f'/static/uploads/{filename}'
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    filename = data.get('filename')
    
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
    
    # TODO: Replace this with your actual CNN model prediction
    # This is mock data for demonstration
    prediction_result = {
        'prediction': 'TB Positive',  # or 'TB Negative'
        'accuracy': '94.5%',
        'precision': '92.3%',
        'recall': '95.8%',
        'f1_score': '94.0%',
        'probability': 0.87,  # 0-1 value for gauge
        'graph_data': {
            'labels': ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 
                      'Class 6', 'Class 7', 'Class 8', 'Class 9', 'Class 10'],
            'values': [85, 92, 78, 88, 95, 82, 90, 87, 93, 79]
        }
    }
    
    return jsonify(prediction_result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)