import tensorflow as tf

# Load saved Keras-model:
original_model = tf.keras.models.load_model("outputs/<run_id>/best_emotion_model.keras")

# Convert to TFLite-format
tflite_model = tf.lite.TFLiteConverter.from_keras_model(original_model).convert()

# Write binary tflite-file
with open("emotion_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model converted and saved as: emotion_model.tflite")

