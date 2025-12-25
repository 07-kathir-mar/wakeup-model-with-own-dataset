import numpy as np
import librosa
import tensorflow as tf
import os
import wave
import time
import serial

# === Load TensorFlow model ===
model_path = r"C:\Users\skath\OneDrive\Desktop\last model\LAST MODEL\LAST_MODEL_TF.keras"
model = tf.keras.models.load_model(model_path)


def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return mfcc_scaled.reshape(1, -1)  # shape: (1, 13)


def predict_audio(file_path):
    features = extract_features(file_path)
    prediction = model.predict(features)
    predicted_label = np.argmax(prediction, axis=1)[0] if prediction.shape[1] > 1 else int(prediction[0] > 0.5)
    return "✅ Correct" if predicted_label == 1 else "❌ Wrong"


# === Recording Setup ===
duration = 3  # seconds
sample_rate = 16000
num_samples = sample_rate * duration
output_folder = r"D:\App\recordings"
serial_port = "COM14"
baud_rate = 921600
bytes_per_sample = 2
expected_bytes = num_samples * bytes_per_sample

# === Prepare Folder & Serial ===
os.makedirs(output_folder, exist_ok=True)
ser = serial.Serial(serial_port, baud_rate, timeout=1)

print(f"🎤 Ready to record {duration}-second clips from ESP32/INMP441.")
print("👉 Press Enter to record...\n")

clip_count = 1

try:
    while True:
        input(f"📢 Press Enter to record clip {clip_count}...")

        ser.reset_input_buffer()
        ser.write(b'R')  # Tell ESP32 to start sending audio

        print("🎙️ Recording...", flush=True)
        start = time.time()

        audio_data = bytearray()
        last_second_shown = 0

        while len(audio_data) < expected_bytes:
            chunk = ser.read(expected_bytes - len(audio_data))
            audio_data.extend(chunk)

            seconds_received = len(audio_data) // (sample_rate * bytes_per_sample)
            if seconds_received > last_second_shown:
                last_second_shown = seconds_received
                print(f"⏱️  Received second: {last_second_shown}")

        end = time.time()

        # Save the clip
        filename = os.path.join(output_folder, f"clip_{clip_count:03d}.wav")
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data)

        print(f"✅ Saved: {filename}")
        print(f"📏 Bytes received: {len(audio_data)} / {expected_bytes}")
        print(f"⏱️  Time taken: {end - start:.2f} seconds")

        # === Predict Immediately ===
        print("🧠 Predicting from recorded audio...")
        result = predict_audio(filename)
        print(f"🔍 Prediction Result: {result}\n")

        if len(audio_data) < expected_bytes:
            print("⚠️ Warning: Incomplete audio received.\n")

        clip_count += 1

except KeyboardInterrupt:
    print("\n🛑 Recording stopped by user.")
finally:
    ser.close()
