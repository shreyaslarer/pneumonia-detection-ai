from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model('best_pneumonia_model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    
    # Read and preprocess image
    img = Image.open(io.BytesIO(file.read())).convert('RGB')
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    prediction = model.predict(img_array)[0][0]
    
    result = {
        'prediction': 'PNEUMONIA' if prediction > 0.5 else 'NORMAL',
        'confidence': float(prediction) if prediction > 0.5 else float(1 - prediction)
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
