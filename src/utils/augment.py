# utils/augment.py — Final Version with pyrubberband time-stretch fix

import numpy as np
import random
import soundfile as sf
import librosa
from librosa.effects import pitch_shift
import pyrubberband as pyrb  # ✅ New import

def load_audio(path, sr=22050, duration=3.0):
    try:
        y, _ = librosa.load(path, sr=sr, duration=duration, res_type='kaiser_fast')
        if len(y) < int(sr * duration):
            pad_width = int(sr * duration) - len(y)
            y = np.pad(y, (0, pad_width))
        return y
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def extract_features(y, sr=22050, n_mels=128, hop_length=512):
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    delta = librosa.feature.delta(mel_db)
    delta2 = librosa.feature.delta(mel_db, order=2)
    return np.stack([mel_db, delta, delta2], axis=0)

def apply_augmentations(y, sr):
    aug_choice = random.choice(['none', 'noise', 'stretch', 'pitch'])

    try:
        if aug_choice == 'noise':
            noise = 0.005 * np.random.randn(len(y))
            y = y + noise

        elif aug_choice == 'stretch':
            rate = random.uniform(0.8, 1.2)
            y = pyrb.time_stretch(y, sr, rate)  # ✅ Proper time-stretch

        elif aug_choice == 'pitch':
            n_steps = random.choice([-2, -1, 1, 2])
            y = pitch_shift(y, sr=sr, n_steps=n_steps)

    except Exception as e:
        print(f"{aug_choice.capitalize()} error: {e}")

    return y

def apply_augmentation_pipeline(audio_path, sr=22050):
    y = load_audio(audio_path, sr=sr)
    if y is None:
        return None

    y_aug = apply_augmentations(y, sr=sr)
    features = extract_features(y_aug, sr=sr)

    # Ensure consistent shape
    if features.shape[2] < 173:
        features = np.pad(features, ((0, 0), (0, 0), (0, 173 - features.shape[2])), mode='constant')
    else:
        features = features[:, :, :173]

    return features.astype(np.float32)
