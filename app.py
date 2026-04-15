from flask import Flask, render_template, request, jsonify
import pickle
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for model
model = None
model_loaded = False

def load_model():
    """Load the sentiment analysis model"""
    global model, model_loaded
    try:
        model_path = 'sentiment_model.pkl'
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return False
        
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        
        model_loaded = True
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def predict_sentiment(text):
    """Predict sentiment for given text"""
    if not model_loaded or model is None:
        return None, "Model not loaded"
    
    try:
        # Validate input
        if not text or not text.strip():
            return None, "Empty text provided"
        
        # Make prediction
        prediction = model.predict([text])[0]
        
        # Get probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba([text])[0]
            class_names = model.classes_
            probabilities = {
                class_names[i]: float(proba[i]) 
                for i in range(len(class_names))
            }
        
        return prediction, None, probabilities
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return None, f"Prediction error: {str(e)}"

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for prediction"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'No text provided',
                'success': False
            }), 400
        
        text = data['text']
        prediction, error, probabilities = predict_sentiment(text)
        
        if error:
            return jsonify({
                'error': error,
                'success': False
            }), 400
        
        response = {
            'prediction': prediction,
            'probabilities': probabilities,
            'text': text,
            'timestamp': datetime.now().isoformat(),
            'success': True
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'success': False
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Load model before starting the app
    if load_model():
        logger.info("Starting Flask application...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        logger.error("Failed to load model. Exiting...")
        exit(1)
