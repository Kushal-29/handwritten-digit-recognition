# 🎯 ULTRA SIMPLE MNIST CNN - RUN DIRECTLY IN VS CODE
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print("🔥 STARTING MNIST PROJECT IN VS CODE...")

# 1. LOAD DATA
print("📥 Loading data...")
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. QUICK LOOK
print(f"Training samples: {len(x_train)}")
print(f"Test samples: {len(x_test)}")

# Show first 10 digits
plt.figure(figsize=(10,3))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
plt.show()

# 3. SIMPLE PREPROCESS
x_train = x_train / 255.0
x_test = x_test / 255.0

# 4. BUILD MODEL (KEEP IT SIMPLE!)
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 5. TRAIN (3 SECONDS!)
print("⚡ Training (3 seconds!)...")
model.fit(x_train, y_train, epochs=5)

# 6. TEST
print("📊 Testing...")
loss, accuracy = model.evaluate(x_test, y_test)
print(f"✅ ACCURACY: {accuracy*100:.2f}%")

# 7. PREDICT SOME DIGITS
predictions = model.predict(x_test[:5])
print("\n🎯 Sample predictions:")
for i in range(5):
    pred = np.argmax(predictions[i])
    actual = y_test[i]
    print(f"Image {i}: Predicted {pred}, Actual {actual} {'✅' if pred==actual else '❌'}")