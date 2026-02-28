#include <iostream> // For printing error messages
#include <string> // for filenames as text
#include <opencv2/opencv.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"


// Function with no return = void
void loadModel(std::string modelPath) {
    // Print for the terminal
    std::cout << "Loading model from: " <<  modelPath << std::endl;
} 

int main(){
    loadModel("emotion_model.tflite");
    return 0;
}

cv::Mat preprocessImage(std::string imagePath){
   
    cv::Mat image = cv::imread(imagePath); //save in image variable
    std::cout << "Loading image from: " << imagePath << std::endl;
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    cv::resize(image, image, cv::Size(48, 48));
    image.convertTo(image, CV_32F, 1.0 / 255.0); //CV_32F bruges til division af pixels
    image = (image - 0.5147) / 0.2536;
    return image;
}

void runInference(std:: string modelPath, cv::Mat image){
    auto model = tflite ::FLatBufferModel::BuildFromFile(modelPath.c_str());
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    interpreter->AllocateTensors();
    std::cout << "Running inference" << std::endl;
}

