import serial
import os
import wave
import time
import sys

# === Settings ===
duration = 3  # seconds
sample_rate = 16000
num_samples = sample_rate * duration
output_folder = r"D:\App\recordings"
serial_port = "COM14"
baud_rate = 921600  # Match ESP32 side

# === Setup ===
os.makedirs(output_folder, exist_ok=True)
ser = serial.Serial(serial_port, baud_rate, timeout=1)

print(f"🎤 Ready to record {duration}-second clips from ESP32/INMP441.")
print("👉 Press Enter to record...\n")

clip_count = 1
bytes_per_sample = 2
expected_bytes = num_samples * bytes_per_sample

try:
    while True:
        input(f"📢 Press Enter to record clip {clip_count}...")

        ser.reset_input_buffer()
        ser.write(b'R')  # Command ESP32 to start recording

        print("🎙️ Recording...", flush=True)
        start = time.time()

        audio_data = bytearray()
        last_second_shown = 0

        while len(audio_data) < expected_bytes:
            chunk = ser.read(expected_bytes - len(audio_data))
            audio_data.extend(chunk)

            # Calculate how many seconds worth of data we received
            seconds_received = len(audio_data) // (sample_rate * bytes_per_sample)
            if seconds_received > last_second_shown:
                last_second_shown = seconds_received
                print(f"⏱️  Received second: {last_second_shown}")

        end = time.time()

        filename = os.path.join(output_folder, f"cipes_{clip_count:03d}.wav")
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data)

        print(f"✅ Saved: {filename}")
        print(f"📏 Bytes received: {len(audio_data)} / {expected_bytes}")
        print(f"⏱️  Time taken: {end - start:.2f} seconds\n")

        if len(audio_data) < expected_bytes:
            print("⚠️ Warning: Incomplete audio received.\n")

        clip_count += 1

except KeyboardInterrupt:
    print("\n🛑 Recording stopped by user.")
finally:
    ser.close()
