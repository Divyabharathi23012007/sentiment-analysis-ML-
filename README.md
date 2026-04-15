# Sentiment Analysis Web Application

A production-ready web application for sentiment analysis using a trained Naïve Bayes model. This application provides an intuitive interface for analyzing text sentiment with real-time predictions and confidence scores.

## Features

- **Real-time Sentiment Analysis**: Instant analysis of text input using a pre-trained Naïve Bayes model
- **Confidence Scores**: Displays probability scores for both positive and negative predictions
- **Modern UI**: Clean, responsive interface built with Tailwind CSS
- **Error Handling**: Comprehensive error handling with user-friendly messages
- **API Endpoints**: RESTful API for programmatic access
- **Health Monitoring**: Built-in health check endpoint
- **Production Ready**: Structured for deployment with proper logging and error handling

## Technology Stack

- **Backend**: Flask (Python web framework)
- **Machine Learning**: Scikit-learn Naïve Bayes with TF-IDF Vectorizer
- **Frontend**: HTML5, Tailwind CSS, JavaScript
- **Model**: Pre-trained sentiment analysis model (sentiment_model.pkl)

## Prerequisites

- Python 3.7 or higher
- pip package manager
- The pre-trained model file: `sentiment_model.pkl`

## Installation

1. **Clone or download the project** to your local machine

2. **Navigate to the project directory**:
   ```bash
   cd sentiment_analysis
   ```

3. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

4. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Verify the model file exists**:
   - Ensure `sentiment_model.pkl` is in the project directory
   - The model should be a scikit-learn Pipeline with TF-IDF Vectorizer and Multinomial Naïve Bayes

## Running the Application

### Development Mode

Start the application in development mode:

```bash
python app.py
```

The application will be available at `http://localhost:5000`

### Production Mode

For production deployment, use Gunicorn:

```bash
# Install Gunicorn (already in requirements.txt)
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## API Documentation

### Predict Endpoint

**POST** `/predict`

Analyzes text sentiment and returns prediction with confidence scores.

**Request Body**:
```json
{
    "text": "Your text to analyze here"
}
```

**Response**:
```json
{
    "prediction": "positive",
    "probabilities": {
        "negative": 0.4627,
        "positive": 0.5373
    },
    "text": "Your text to analyze here",
    "timestamp": "2024-01-15T10:30:00.000Z",
    "success": true
}
```

**Error Response**:
```json
{
    "error": "Error message",
    "success": false
}
```

### Health Check Endpoint

**GET** `/health`

Returns the application status and model loading state.

**Response**:
```json
{
    "status": "healthy",
    "model_loaded": true,
    "timestamp": "2024-01-15T10:30:00.000Z"
}
```

## Usage

1. **Open the application** in your web browser at `http://localhost:5000`

2. **Enter or paste text** in the input area

3. **Click "Analyze Sentiment"** or press `Ctrl+Enter`

4. **View results** including:
   - Predicted sentiment (Positive/Negative)
   - Confidence scores with visual progress bars
   - Original text analysis
   - Timestamp of analysis

5. **Use additional features**:
   - Character counter for input text
   - Copy results to clipboard
   - Start new analysis
   - Real-time status indicators

## Model Information

The application uses a pre-trained scikit-learn Pipeline consisting of:

1. **TF-IDF Vectorizer**: Converts text to numerical features
2. **Multinomial Naïve Bayes**: Performs sentiment classification

The model was trained in Google Colab and saved using Pickle. It classifies text into two categories:
- **Positive**: Text with positive sentiment
- **Negative**: Text with negative sentiment

## Project Structure

```
sentiment_analysis/
    app.py                 # Main Flask application
    sentiment_model.pkl    # Pre-trained model file
    requirements.txt       # Python dependencies
    templates/            # HTML templates
        index.html        # Main application interface
        404.html          # 404 error page
        500.html          # 500 error page
    test_model.py         # Model testing script (optional)
    README.md            # This documentation
```

## Error Handling

The application includes comprehensive error handling:

- **Model Loading Errors**: Graceful handling if model file is missing or corrupted
- **Input Validation**: Validates text input before processing
- **Prediction Errors**: Handles errors during model inference
- **Network Errors**: Provides feedback for API connectivity issues
- **User-friendly Messages**: Clear error messages for end users

## Logging

The application uses Python's logging module to track:
- Model loading status
- Prediction requests and results
- Error conditions
- Application startup

Logs are output to the console and can be configured for file output in production.

## Deployment Considerations

For production deployment:

1. **Use a production WSGI server** like Gunicorn or uWSGI
2. **Configure proper logging** to files
3. **Set up environment variables** for configuration
4. **Use a reverse proxy** like Nginx for SSL termination
5. **Implement rate limiting** for API endpoints
6. **Set up monitoring** and alerting
7. **Regular backups** of the model file

## Troubleshooting

### Common Issues

1. **Model not loading**:
   - Verify `sentiment_model.pkl` exists in the project directory
   - Check if the model file is corrupted
   - Ensure all required dependencies are installed

2. **Port already in use**:
   - Change the port in the `app.run()` command
   - Kill existing processes using the port

3. **Dependencies not installing**:
   - Upgrade pip: `pip install --upgrade pip`
   - Try installing packages individually
   - Check Python version compatibility

### Testing the Model

Use the provided test script to verify model functionality:

```bash
python test_model.py
```

This will load the model and perform a test prediction.

## Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

For issues and questions:
- Check the troubleshooting section
- Review the application logs
- Verify all prerequisites are met
- Test the model independently using `test_model.py`

---

**Note**: This application requires the `sentiment_model.pkl` file to function. Ensure the model file is properly trained and compatible with the scikit-learn version specified in requirements.txt.
