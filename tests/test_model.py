# 🧪 TEST MODEL WITHOUT DRAWING
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

print("🧪 Testing MNIST Model")
print("="*40)

# Load model
model = tf.keras.models.load_model('my_mnist_model.h5')

# Create test digits programmatically
digits = []

# Digit 0 (circle)
img0 = np.zeros((28, 28))
for i in range(28):
    for j in range(28):
        if 6 < i < 22 and 6 < j < 22:
            if abs(14-i)**2 + abs(14-j)**2 < 64:
                img0[i, j] = 1

# Digit 1 (vertical line)
img1 = np.zeros((28, 28))
img1[5:23, 14] = 1

# Digit 7
img7 = np.zeros((28, 28))
img7[5, 5:23] = 1  # Top horizontal
for i in range(6, 23):
    img7[i, 23-i] = 1  # Diagonal

# Test predictions
test_images = [img0, img1, img7]
expected = [0, 1, 7]

print("\n🎯 Testing model predictions:")
for idx, (img, expected_digit) in enumerate(zip(test_images, expected)):
    img_array = img.reshape(1, 28, 28, 1)
    prediction = model.predict(img_array, verbose=0)
    predicted = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    
    print(f"\nDigit {expected_digit}:")
    print(f"  Predicted: {predicted} {'✅' if predicted == expected_digit else '❌'}")
    print(f"  Confidence: {confidence:.1f}%")
    
    # Show image
    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap='gray')
    plt.title(f"Digit {expected_digit} → Predicted: {predicted}")
    plt.axis('off')
    plt.show()

print("\n" + "="*40)
print("✅ Model test complete!")