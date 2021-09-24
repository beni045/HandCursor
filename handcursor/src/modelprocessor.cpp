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


#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

ModelProcessor::ModelProcessor(int orig_width, int orig_height, int resize_width, int resize_height, std::string model_path)
:
orig_width_(orig_width),
orig_height_(orig_height),
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
    TFLITE_MINIMAL_CHECK(interpreter_->Invoke() == kTfLiteOk);
}

void ModelProcessor::Postprocess(){
}

void ModelProcessor::Process(cv::Mat orig_image){
    orig_image_ = orig_image;
    Preprocess();
    Inference();
    Postprocess();
}