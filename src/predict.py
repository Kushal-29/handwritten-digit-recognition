import numpy as np
from tensorflow import keras
import cv2

def predict_digit(model, image):
    """
    Predict digit from a single image
    """
    # Preprocess image (same as training)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    image = cv2.resize(image, (28, 28))
    image = image.reshape(1, 28, 28, 1).astype('float32') / 255
    
    # Predict
    prediction = model.predict(image, verbose=0)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction)
    
    return predicted_digit, confidence, prediction[0]

def preprocess_custom_image(image_path):
    """
    Preprocess custom image for prediction
    """
    # Read and preprocess
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Invert if needed (MNIST digits are white on black)
    img = 255 - img
    
    # Threshold
    _, img = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
    
    # Resize
    img = cv2.resize(img, (28, 28))
    
    # Reshape for model
    img = img.reshape(1, 28, 28, 1).astype('float32') / 255
    
    return img