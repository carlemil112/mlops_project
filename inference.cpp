#include <iostream> // For printing error messages
#include <string> // for filenames as text
// TFLite header gets added later :D


// Function with no return = void
void loadModel(std::string modelPath) {
    // Print for the terminal
    std::cout << "Loading model from: " <<  modelPath << std::endl;
} 

int main(){
    loadModel("emotion_model.tflite");
    return 0;
}