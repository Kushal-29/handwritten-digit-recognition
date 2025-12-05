import streamlit as st
import numpy as np
import cv2
from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image
import io

# Page config
st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    return keras.models.load_model('models/mnist_cnn.h5')

# Main app
def main():
    st.title("🔢 Handwritten Digit Recognition")
    st.markdown("Draw a digit (0-9) and the CNN model will predict it!")
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This app uses a Convolutional Neural Network (CNN) trained on the MNIST dataset.
        
        **Model Performance:**
        - Accuracy: >98%
        - Architecture: CNN with 3 Conv layers
        - Framework: TensorFlow/Keras
        """)
        
        st.header("How to Use")
        st.markdown("""
        1. Draw a digit in the canvas
        2. Click 'Predict'
        3. See the model's prediction
        """)
    
    # Main content - two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Draw Your Digit")
        
        # Canvas for drawing
        canvas_result = st.canvas(
            stroke_width=15,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas"
        )
        
        # Buttons
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            predict_btn = st.button("🎯 Predict", use_container_width=True)
        with col_btn2:
            clear_btn = st.button("🗑️ Clear", use_container_width=True)
        with col_btn3:
            sample_btn = st.button("📊 Show Sample", use_container_width=True)
    
    with col2:
        st.subheader("🤖 Model Prediction")
        
        # Initialize session state
        if 'prediction' not in st.session_state:
            st.session_state.prediction = None
        if 'confidence' not in st.session_state:
            st.session_state.confidence = None
        
        # Load model
        try:
            model = load_model()
        except:
            st.error("Model not found! Please train the model first.")
            st.info("Run: `python src/train.py`")
            return
        
        # Prediction logic
        if predict_btn and canvas_result is not None:
            # Convert canvas image to numpy array
            img = cv2.cvtColor(np.array(canvas_result.image_data), cv2.COLOR_RGBA2GRAY)
            
            # Preprocess
            img = cv2.resize(img, (28, 28))
            img = img.reshape(1, 28, 28, 1).astype('float32') / 255
            
            # Predict
            prediction = model.predict(img, verbose=0)
            predicted_digit = np.argmax(prediction)
            confidence = np.max(prediction)
            
            # Store in session state
            st.session_state.prediction = predicted_digit
            st.session_state.confidence = confidence
            
            # Display result
            st.markdown(f"## Prediction: **{predicted_digit}**")
            st.markdown(f"### Confidence: **{confidence:.2%}**")
            
            # Confidence bar
            st.progress(float(confidence))
            
            # Show all probabilities
            st.subheader("All Probabilities:")
            probs = prediction[0]
            
            # Create bar chart
            fig, ax = plt.subplots(figsize=(10, 4))
            bars = ax.bar(range(10), probs, color='skyblue')
            bars[predicted_digit].set_color('red')
            ax.set_xlabel('Digit')
            ax.set_ylabel('Probability')
            ax.set_title('Model Confidence for Each Digit')
            ax.set_xticks(range(10))
            ax.set_ylim([0, 1])
            
            # Add value labels
            for i, v in enumerate(probs):
                ax.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
            
            st.pyplot(fig)
        
        elif clear_btn:
            st.session_state.prediction = None
            st.session_state.confidence = None
            st.rerun()
        
        elif sample_btn:
            st.subheader("Sample Predictions")
            st.image("training_history.png", caption="Training Progress", use_column_width=True)
        
        # Show stored prediction if exists
        elif st.session_state.prediction is not None:
            st.markdown(f"## Last Prediction: **{st.session_state.prediction}**")
            st.markdown(f"### Confidence: **{st.session_state.confidence:.2%}**")
            st.info("Draw a new digit and click 'Predict' for new prediction.")
    
    # Footer
    st.markdown("---")
    st.markdown("**Built with TensorFlow, Keras, and Streamlit** | *MNIST Digit Recognition*")

if __name__ == "__main__":
    main()