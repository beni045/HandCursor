#include <iostream>
#include <opencv2/highgui.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "utils.h"
#pragma once

class ModelProcessor{       
    protected:            
        // Image processing
        int orig_width_, orig_height_;  
        const int resize_width_, resize_height_;  

        cv::Mat orig_image_;    

        float* input_tensor_;
        
        float* output_tensor1_;
        float* output_tensor2_;

        // Inference model
        std::unique_ptr<tflite::FlatBufferModel> model_;
        tflite::ops::builtin::BuiltinOpResolver resolver_;
        std::unique_ptr<tflite::Interpreter> interpreter_;


    public:
        ModelProcessor(int resize_width, int resize_height, std::string filname);
        ~ModelProcessor();
        
        int8_t Process(cv::Mat orig_image);
   
    protected:
        virtual void Preprocess() = 0;
        void Inference();
        virtual int8_t Postprocess() = 0;
        
};