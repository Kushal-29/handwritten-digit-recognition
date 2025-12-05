from tensorflow import keras
from tensorflow.keras import layers

def create_cnn_model():
    """
    Create CNN model for MNIST digit recognition
    """
    model = keras.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Flatten and Dense Layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),  # Regularization
        layers.Dense(10, activation='softmax')  # Output layer for 10 digits
    ])
    
    return model

def compile_model(model):
    """
    Compile the CNN model
    """
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def model_summary(model):
    """
    Print model summary
    """
    model.summary()