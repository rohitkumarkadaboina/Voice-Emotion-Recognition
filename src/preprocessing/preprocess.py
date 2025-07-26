import os
import librosa
import numpy as np
import random

from src.utils.augment import add_noise, time_stretch, pitch_shift

emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def extract_features_from_audio(audio, sr, n_mfcc=40, max_len=173):
    # MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc = librosa.util.fix_length(mfcc, size=max_len, axis=1)

    # Chroma
    stft = np.abs(librosa.stft(audio))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
    chroma = librosa.util.fix_length(chroma, size=max_len, axis=1)

    # Spectral Contrast
    contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
    contrast = librosa.util.fix_length(contrast, size=max_len, axis=1)

    # Stack all features vertically: (n_features, max_len)
    combined = np.vstack([mfcc, chroma, contrast])

    return combined  # Shape: (40 + 12 + 7, max_len) = (59, 173)


def load_data(data_dir):
    X, y = [], []

    print(f"Scanning: {data_dir}")
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                print("Found:", file_path)

                try:
                    emotion_id = file.split("-")[2]
                    emotion = emotion_map.get(emotion_id)

                    if emotion:
                        audio, sr = librosa.load(file_path, res_type='kaiser_fast')
                        features = extract_features_from_audio(audio, sr)
                        X.append(features)
                        y.append(emotion)

                        # Apply one random augmentation
                        aug_type = random.choice(['noise', 'stretch', 'pitch'])
                        if aug_type == 'noise':
                            aug_audio = add_noise(audio)
                        elif aug_type == 'stretch':
                            aug_audio = time_stretch(audio)
                        elif aug_type == 'pitch':
                            aug_audio = pitch_shift(audio, sr)

                        aug_features = extract_features_from_audio(aug_audio, sr)
                        X.append(aug_features)
                        y.append(emotion)

                except Exception as e:
                    print(f"Skipping file {file} due to error: {e}")

    return np.array(X), np.array(y)


if __name__ == "__main__":
    data_path = "data/ravdess/"
    X, y = load_data(data_path)
    print("Feature shape:", X.shape)
    print("Labels shape:", y.shape)
    print("Sample labels:", y[:10])
    np.save("features.npy", X)
    np.save("labels.npy", y)

