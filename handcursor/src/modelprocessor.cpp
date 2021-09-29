#include <modelprocessor.h>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include <iostream>
#include <fstream>
#include <cstdio>
#include <chrono>


#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

ModelProcessor::ModelProcessor(int resize_width, int resize_height, std::string model_path)
:
resize_width_(resize_width),
resize_height_(resize_height)
{
    model_ = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    TFLITE_MINIMAL_CHECK(model_ != nullptr);

    tflite::InterpreterBuilder(*model_, resolver_)(&interpreter_);

    TFLITE_MINIMAL_CHECK(interpreter_ != nullptr);
    TFLITE_MINIMAL_CHECK(interpreter_->AllocateTensors() == kTfLiteOk);

    input_tensor_ = interpreter_->typed_input_tensor<float>(0);
    output_tensor1_ = interpreter_->typed_output_tensor<float>(0);
    output_tensor2_ = interpreter_->typed_output_tensor<float>(1);
}

ModelProcessor::~ModelProcessor(){
}

void ModelProcessor::Preprocess(){
}

void ModelProcessor::Inference(){
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    begin = std::chrono::steady_clock::now();
    TFLITE_MINIMAL_CHECK(interpreter_->Invoke() == kTfLiteOk);
    end = std::chrono::steady_clock::now();
    std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;
}

int8_t ModelProcessor::Postprocess(){
    return SUCCESS;     // this shouldnt have to be defined?
}

int8_t ModelProcessor::Process(cv::Mat orig_image){
    orig_image_ = orig_image;
    orig_width_ = orig_image.cols;
    orig_height_ = orig_image.rows;
    Preprocess();
    Inference();
    int8_t status = ERROR_CHECK(Postprocess());
    return status;
}