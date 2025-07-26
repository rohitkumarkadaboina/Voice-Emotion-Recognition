# Improve the Model (Version 2)
# 🔧 What We’ll Do:
# Make the model deeper: More Conv1D + BatchNorm + Dropout

# Use Bidirectional LSTM to better capture emotion flow

# Stratified split so all emotion classes are balanced

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dropout, BatchNormalization
from tensorflow.keras.layers import Bidirectional, LSTM, Dense

# Load features
X = np.load("features.npy")
y = np.load("labels.npy")

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Train-test split with stratify
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
)

# Build better model
model = Sequential([
    Input(shape=(40, 173)),

    Conv1D(64, 5, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.3),

    Conv1D(128, 5, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.3),

    Bidirectional(LSTM(64)),
    Dropout(0.4),

    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(y_categorical.shape[1], activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

model.save("emotion_model_v2.keras")

test_loss, test_acc = model.evaluate(X_test, y_test)
print("Improved Test accuracy:", test_acc)


