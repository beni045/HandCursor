#include <iostream>
#include <opencv2/highgui.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#define ANCHORS_LENGTH 2944

class HandDetector{       
    private:            
        // Image processing
        const int orig_width_, orig_height_;  
        const int resize_width_, resize_height_;  

        cv::Mat orig_image_;    

        float* input_tensor_;
        float* output_tensor1_;
        float* output_tensor2_;

        std::vector<float> anchors_;




        // Inference model
        std::unique_ptr<tflite::FlatBufferModel> model_;
        tflite::ops::builtin::BuiltinOpResolver resolver_;
        std::unique_ptr<tflite::Interpreter> interpreter_;


    public:
        HandDetector(int orig_width, int orig_height, int resize_width, int resize_height, std::string filname);
        ~HandDetector();
        
        std::vector<float> LoadAnchors(std::string filepath);

        void Preprocess();
        void Inference();
        void Postprocess();

        int FindWidest(std::vector<int> threshold_idxs);

        cv::Rect FindBbox(int widest_idx);
        std::vector<cv::Point> FindKeypoints(int widest_idx);
        cv::Mat TransformPalm(cv::Point wrist, cv::Point middlefinger, float thirdpoint_scale);

        void DrawBboxOrig(cv::Rect rectangle);

        void Process(cv::Mat orig_image);




    // Inference();
    // Postprocess();

    // input: frame
    // output: cropped frame


    // STEPS
    // resize orig_img
    // regularize resized_img
    // format/copy resized_img to model input
    // proc inference
    // postprocess output

};


