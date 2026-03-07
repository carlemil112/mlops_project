#include <iostream>  // For printing error messages
#include <string>    // For filenames as text
#include <opencv2/opencv.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

// Block 1: Load model - prints the path we are loading from
void loadModel(std::string modelPath) {
    std::cout << "Loading model from: " << modelPath << std::endl;
}

// Block 2: Load and preprocess image
cv::Mat preprocessImage(std::string imagePath) {
    // Load image from disk into cv::Mat (2D pixel array)
    cv::Mat image = cv::imread(imagePath);
    std::cout << "Loading image from: " << imagePath << std::endl;

    // Convert from BGR (OpenCV default) to grayscale (1 channel)
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

    // Resize to 48x48 pixels (model input size)
    cv::resize(image, image, cv::Size(48, 48));

    // Convert pixels to 32-bit float and scale from [0, 255] to [0, 1]
    image.convertTo(image, CV_32F, 1.0 / 255.0);

    // Normalize using dataset mean and std (same as training)
    image = (image - 0.5147) / 0.2536;

    return image;
}

// Block 3: Run inference - loads model, copies pixels, runs model, returns 7 probabilities
float* runInference(std::string modelPath, cv::Mat image) {
    std::cout << "Running inference" << std::endl;

    // Load .tflite model file from disk
    auto model = tflite::FlatBufferModel::BuildFromFile(modelPath.c_str());

    // Resolver: list of all mathematical operations TFLite knows
    tflite::ops::builtin::BuiltinOpResolver resolver;

    // Interpreter: the engine that runs the model
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);

    // Allocate memory for input and output tensors
    interpreter->AllocateTensors();

    // Copy preprocessed image pixels into model input buffer
    float* input = interpreter->typed_input_tensor<float>(0);
    float* imageData = (float*)image.data;
    memcpy(input, imageData, 48 * 48 * sizeof(float));

    // Run the model
    interpreter->Invoke();

    // Read output buffer (7 probabilities, one per emotion)
    float* output = interpreter->typed_output_tensor<float>(0);
    return output;
}

// Block 4: Format and print results as JSON to stdout
void printJSON(float* output) {
    // Array of emotion labels - same order as model output
    std::string emotions[] = {"angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"};

    // Print opening bracket for JSON
    std::cout << "{" << std::endl;

    // Loop through all 7 emotions
    for (int i = 0; i < 7; i++) {
        // Print emotion label in quotes + probability from output buffer
        std::cout << "    \"" << emotions[i] << "\": " << output[i];

        // Comma after every line except the last
        if (i < 6) std::cout << ",";

        // New line after each emotion
        std::cout << std::endl;
    }

    // Print closing bracket for JSON
    std::cout << "}" << std::endl;
}

// Entry point - ties all blocks together
int main() {
    loadModel("emotion_model.tflite");
    cv::Mat image = preprocessImage("emotion_model.tflite");
    float* output = runInference("emotion_model.tflite", image);
    printJSON(output);
    return 0;
}
