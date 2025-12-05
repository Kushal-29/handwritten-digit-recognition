# 🎨 WORKING DRAWING APP - FIXED IMAGE ISSUE
import tkinter as tk
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
import os

print("🚀 Starting Drawing App...")
print("="*50)

# 1. LOAD OR CREATE MODEL
if not os.path.exists('my_mnist_model.h5'):
    print("⚠️ No model found! Training a quick one...")
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28*28) / 255.0
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=0)
    model.save('my_mnist_model.h5')
    print("✅ Quick model trained and saved!")
else:
    print("✅ Found existing model!")

# Load model
model = tf.keras.models.load_model('my_mnist_model.h5')
print("🤖 Model loaded successfully!")

# 2. CREATE MAIN WINDOW
window = tk.Tk()
window.title("✏️ Draw Digit - AI Predicts")
window.geometry("500x600")
window.configure(bg='#f0f0f0')

# Title
title = tk.Label(window, text="✏️ HANDWRITTEN DIGIT RECOGNITION", 
                font=("Arial", 16, "bold"),
                bg='#f0f0f0', fg='#2c3e50')
title.pack(pady=10)

# Instructions
instructions = tk.Label(window, 
                       text="1. Draw a digit (0-9) in the black box\n2. Click PREDICT\n3. See AI prediction",
                       font=("Arial", 11),
                       bg='#f0f0f0', fg='#34495e')
instructions.pack()

# Canvas for drawing
canvas_frame = tk.Frame(window, bg='#f0f0f0')
canvas_frame.pack(pady=10)

canvas = tk.Canvas(canvas_frame, width=280, height=280, 
                  bg='black', cursor="cross")
canvas.pack()

# Result display
result_frame = tk.Frame(window, bg='#f0f0f0')
result_frame.pack(pady=20)

result_text = tk.StringVar()
result_text.set("Draw a digit and click PREDICT")

result_label = tk.Label(result_frame, textvariable=result_text,
                       font=("Arial", 18, "bold"),
                       bg='#f0f0f0', fg='#3498db')
result_label.pack()

confidence_text = tk.StringVar()
confidence_text.set("Confidence: --%")

confidence_label = tk.Label(result_frame, textvariable=confidence_text,
                           font=("Arial", 14),
                           bg='#f0f0f0', fg='#2c3e50')
confidence_label.pack()

# Buttons
button_frame = tk.Frame(window, bg='#f0f0f0')
button_frame.pack(pady=10)

# 3. DRAWING LOGIC
class DrawingApp:
    def __init__(self):
        self.image = Image.new("L", (280, 280), 0)  # Black image
        self.draw = ImageDraw.Draw(self.image)
        self.last_x = None
        self.last_y = None
        
    def start_draw(self, event):
        self.last_x = event.x
        self.last_y = event.y
        
    def draw_line(self, event):
        if self.last_x and self.last_y:
            # Draw on canvas
            canvas.create_line(self.last_x, self.last_y, 
                             event.x, event.y, 
                             width=15, fill='white',
                             capstyle=tk.ROUND, smooth=True)
            
            # Draw on PIL image
            self.draw.line([self.last_x, self.last_y, event.x, event.y], 
                         fill=255, width=15)
        
        self.last_x = event.x
        self.last_y = event.y
        
    def end_draw(self, event):
        self.last_x = None
        self.last_y = None
        
    def clear(self):
        canvas.delete("all")
        self.image = Image.new("L", (280, 280), 0)
        self.draw = ImageDraw.Draw(self.image)
        result_text.set("Draw a digit and click PREDICT")
        confidence_text.set("Confidence: --%")
        print("🗑️ Canvas cleared")
        
    def predict(self):
        try:
            print("\n🔍 Making prediction...")
            
            # Convert to MNIST format (28x28)
            img_small = self.image.resize((28, 28))
            
            # Debug: Save the image to see what we're sending
            img_small.save("debug_digit.png")
            print("💾 Saved drawing as 'debug_digit.png'")
            
            # Convert to numpy array
            img_array = np.array(img_small)
            
            # Check if image is not empty
            if img_array.max() == 0:
                result_text.set("❌ Please draw something!")
                return
                
            # Reshape and normalize
            img_array = img_array.reshape(1, 28, 28, 1).astype('float32') / 255.0
            
            print(f"📊 Image shape: {img_array.shape}")
            print(f"📊 Pixel range: {img_array.min():.3f} to {img_array.max():.3f}")
            
            # Make prediction
            prediction = model.predict(img_array, verbose=0)
            digit = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            
            print(f"✅ Prediction complete!")
            print(f"🎯 Predicted digit: {digit}")
            print(f"📈 Confidence: {confidence:.1f}%")
            
            # Show all probabilities
            print("\n📊 All probabilities:")
            for i in range(10):
                prob = prediction[0][i] * 100
                print(f"  {i}: {prob:5.1f}%")
            
            # Update UI
            result_text.set(f"🎯 AI Predicts: {digit}")
            confidence_text.set(f"Confidence: {confidence:.1f}%")
            
            # Color based on confidence
            if confidence > 90:
                result_label.config(fg='#27ae60')  # Green
            elif confidence > 70:
                result_label.config(fg='#f39c12')  # Orange
            else:
                result_label.config(fg='#e74c3c')  # Red
                
        except Exception as e:
            print(f"❌ Error: {e}")
            result_text.set("❌ Prediction failed")
            confidence_text.set("Check console for error")

# Create drawing instance
drawing = DrawingApp()

# Bind events
canvas.bind("<Button-1>", drawing.start_draw)
canvas.bind("<B1-Motion>", drawing.draw_line)
canvas.bind("<ButtonRelease-1>", drawing.end_draw)

# Create buttons
predict_btn = tk.Button(button_frame, text="🎯 PREDICT", 
                       font=("Arial", 12, "bold"),
                       bg="#2ecc71", fg="white",
                       width=12, height=2,
                       command=drawing.predict)
predict_btn.pack(side='left', padx=5)

clear_btn = tk.Button(button_frame, text="🗑️ CLEAR", 
                     font=("Arial", 12),
                     bg="#e74c3c", fg="white",
                     width=12, height=2,
                     command=drawing.clear)
clear_btn.pack(side='left', padx=5)

# Test button - draws a sample 7
def draw_sample_7():
    drawing.clear()
    # Draw a 7 on canvas
    canvas.create_line(50, 50, 230, 50, width=15, fill='white')  # Top
    canvas.create_line(200, 50, 80, 230, width=15, fill='white')  # Diagonal
    
    # Draw on PIL image
    drawing.draw.line([50, 50, 230, 50], fill=255, width=15)
    drawing.draw.line([200, 50, 80, 230], fill=255, width=15)
    print("✅ Drew sample digit '7'")

test_btn = tk.Button(button_frame, text="7️⃣ TEST 7", 
                    font=("Arial", 12),
                    bg="#3498db", fg="white",
                    width=12, height=2,
                    command=draw_sample_7)
test_btn.pack(side='left', padx=5)

# Footer
footer = tk.Label(window, 
                 text="Draw digits like: 0 1 2 3 4 5 6 7 8 9",
                 font=("Arial", 10),
                 bg='#f0f0f0', fg='#7f8c8d')
footer.pack(pady=20)

print("="*50)
print("✅ App ready! Draw a digit and click PREDICT")
print("="*50)

# Start app
window.mainloop()