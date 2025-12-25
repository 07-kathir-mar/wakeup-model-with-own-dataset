import os
import numpy as np
import librosa
import pickle

# ✅ Your dataset paths  "C:\Users\skath\OneDrive\Desktop\last model\LAST DATASET\correct"   "C:\Users\skath\OneDrive\Desktop\last model\LAST DATASET\wrong"
correct_dir = r"C:\Users\skath\OneDrive\Desktop\last model\LAST DATASET\correct"
wrong_dir = r"C:\Users\skath\OneDrive\Desktop\last model\LAST DATASET\wrong"

# ✅ Output path  "C:\Users\skath\OneDrive\Desktop\last model\WAV TO PKL CONVERTED"
OUTPUT_PATH = r"C:\Users\skath\OneDrive\Desktop\last model\WAV TO PKL CONVERTED"
os.makedirs(OUTPUT_PATH, exist_ok=True)  # Create output folder if it doesn't exist

# 🎯 Function to extract MFCC features
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)  # Load with 16 kHz sampling rate
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return mfcc_scaled

# ✅ Lists for storing features and labels
features = []
labels = []

# ✅ Extract features from correct audio files (label = 1)
for filename in os.listdir(correct_dir):
    if filename.endswith(".wav"):
        path = os.path.join(correct_dir, filename)
        mfcc = extract_features(path)
        features.append(mfcc)
        labels.append(1)

# ✅ Extract features from wrong audio files (label = 0)
for filename in os.listdir(wrong_dir):
    if filename.endswith(".wav"):
        path = os.path.join(wrong_dir, filename)
        mfcc = extract_features(path)
        features.append(mfcc)
        labels.append(0)

# ✅ Convert lists to numpy arrays
X = np.array(features)
y = np.array(labels)

# ✅ Save to file
output_file = os.path.join(OUTPUT_PATH, "audio_features.pkl")
with open(output_file, "wb") as f:
    pickle.dump((X, y), f)

print(f"✅ MFCC Extraction Done. Saved to: {output_file}")
