import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def load_trained_model(model_path='models/mnist_cnn.h5'):
    """
    Load trained model
    """
    return keras.models.load_model(model_path)

def evaluate_model(model, test_images, test_labels):
    """
    Evaluate model and generate metrics
    """
    # Predict
    predictions = model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(test_labels, axis=1)
    
    # Confusion Matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # Classification Report
    report = classification_report(true_labels, predicted_labels, target_names=[str(i) for i in range(10)])
    
    return predicted_labels, cm, report

def plot_confusion_matrix(cm):
    """
    Plot confusion matrix
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[str(i) for i in range(10)], 
                yticklabels=[str(i) for i in range(10)])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

def plot_sample_predictions(test_images, true_labels, predicted_labels, num_samples=12):
    """
    Plot sample predictions
    """
    plt.figure(figsize=(12, 8))
    for i in range(num_samples):
        plt.subplot(3, 4, i+1)
        plt.imshow(test_images[i].reshape(28, 28), cmap='gray')
        
        color = 'green' if true_labels[i] == predicted_labels[i] else 'red'
        plt.title(f"True: {true_labels[i]}\nPred: {predicted_labels[i]}", color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()