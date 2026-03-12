import torch
from torch import onnx
import tensorflow as tf
from train import FERModel
import onnx_tf

# path from best model
model_path = "outputs/fer_run/best_emotion_model.pt"

# Load PYTORCH model - NOT keras :D
model = FERModel(7)
model.load_state_dict(torch.load(model_path))
model.eval()

# Export to ONNX 
dummy_input = torch.zeros((1, 1, 48,48)) #Probably needs dummy input to see if it works first? I'm guessing the bigger model would take longer
onnx.export(model, dummy_input, "emotion_model.onnx")

# Convert ONNX to TensorFlow
onnx_model = onnx.load("emotion_model.onnx")
tf_model = onnx_tf.backend.prepare(onnx_model)
tf_model.export_graph("emotion_model_tf")

# Convert from tf to tflite
tflite_model = tf.lite.TFLiteConverter.from_saved_model("emotion_model_tf").convert()

# Save .tflite file
with open("emotion_model.tflite", "wb") as f:
    f.write(tflite_model)


