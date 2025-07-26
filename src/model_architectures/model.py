import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten

# Load preprocessed data
X = np.load("features.npy")  # We'll save this in a bit
y = np.load("labels.npy")

# Encode string labels into numbers
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42
)

# Model architecture
model = Sequential()
model.add(Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(40, 173)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))

model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.3))

model.add(Dense(64, activation='relu'))
model.add(Dense(y_categorical.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# Save model
model.save("emotion_model.h5")

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test accuracy:", test_acc)

# Your model is learning something, but:

# It's overfitting (high train acc, low test acc)

# The model may be too simple or not generalized enough

# Dataset is imbalanced (some emotions like neutral or calm dominate)

# 🧠 What Can You Do to Improve?
# Let’s try the following next steps:

# ✅ 1. Model Improvements
# Add more Conv1D + Pooling layers

# Add Batch Normalization

# Try Bidirectional LSTM instead of plain LSTM

# ✅ 2. Use Data Augmentation
# Pitch shifting

# Time stretching

# Background noise

# (This helps make the model robust to real-world voices!)

# ✅ 3. Use Stratified Train-Test Split
# To ensure emotion classes are balanced during training.

# ✅ 4. Tune Hyperparameters
# Try different batch_size, epochs, dropout, etc.
