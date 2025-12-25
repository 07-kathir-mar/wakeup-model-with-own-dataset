import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# === Load the features and labels ===
data_path = r"C:\Users\skath\OneDrive\Desktop\last model\WAV TO PKL CONVERTED\audio_features.pkl"
with open(data_path, "rb") as f:
    X, y = pickle.load(f)

# === Encode labels as integers ===
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(np.unique(y_encoded))

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# === Compute class weights ===
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

# === Convert to numpy arrays ===
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# === Define a simple Neural Network model ===
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# === Train the model with class weights ===
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, class_weight=class_weight_dict)

# === Evaluate on test data ===
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Model Test Accuracy: {acc * 100:.2f}%")

# === Save the model ===
model_path = r"C:\Users\skath\OneDrive\Desktop\last model\LAST MODEL\LAST_MODEL_TF.keras"
model.save(model_path)
print(f"✅ TensorFlow model saved at: {model_path}")
