import os
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from data_preprocessing import load_data, preprocess_data, prepare_labels
from model import create_cnn_model, compile_model

def train_model():
    """
    Main training function
    """
    print("🚀 Starting MNIST Digit Recognition Training...")
    
    # 1. Load data
    print("📥 Loading MNIST dataset...")
    (train_images, train_labels), (test_images, test_labels) = load_data()
    
    # 2. Preprocess data
    print("🔧 Preprocessing data...")
    train_images, test_images = preprocess_data(train_images, test_images)
    train_labels, test_labels = prepare_labels(train_labels, test_labels)
    
    # 3. Create model
    print("🧠 Creating CNN model...")
    model = create_cnn_model()
    model = compile_model(model)
    
    # 4. Train model
    print("🏋️ Training model...")
    history = model.fit(
        train_images, train_labels,
        epochs=10,
        batch_size=64,
        validation_split=0.2,
        verbose=1
    )
    
    # 5. Evaluate model
    print("📊 Evaluating model...")
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    print(f"\n✅ Model Evaluation:")
    print(f"   Test Accuracy: {test_acc:.4f}")
    print(f"   Test Loss: {test_loss:.4f}")
    
    # 6. Save model
    print("💾 Saving model...")
    os.makedirs('models', exist_ok=True)
    model.save('models/mnist_cnn.h5')
    print("✅ Model saved as 'models/mnist_cnn.h5'")
    
    # 7. Plot training history
    plot_training_history(history)
    
    return model, history, test_acc, test_loss

def plot_training_history(history):
    """
    Plot training and validation accuracy/loss
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    train_model()