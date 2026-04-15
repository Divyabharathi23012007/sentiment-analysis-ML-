import pickle
import numpy as np

# Load and examine the model
try:
    with open('sentiment_model.pkl', 'rb') as file:
        model_data = pickle.load(file)
    
    print("Model type:", type(model_data))
    print("Pipeline steps:")
    for name, step in model_data.steps:
        print(f"  {name}: {type(step)}")
        if hasattr(step, '__dict__'):
            print(f"    Attributes: {list(step.__dict__.keys())}")
    
    if hasattr(model_data, 'classes_'):
        print("Classes:", model_data.classes_)
    
    # Test prediction
    test_text = "This is a great movie!"
    try:
        prediction = model_data.predict([test_text])
        print(f"Test prediction for '{test_text}': {prediction}")
        
        # Get prediction probabilities
        if hasattr(model_data, 'predict_proba'):
            probabilities = model_data.predict_proba([test_text])
            print(f"Probabilities: {probabilities}")
    except Exception as pred_error:
        print(f"Prediction error: {pred_error}")
    
    print("Model loaded successfully!")
    
except Exception as e:
    print(f"Error loading model: {e}")
