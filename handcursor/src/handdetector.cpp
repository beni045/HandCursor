#include <handdetector.h>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"


HandDetector::HandDetector(int orig_width, int orig_height, int resize_width, int resize_height, std::string palm_detector_path)
:
orig_width_(orig_width),
orig_height_(orig_height),
resize_width_(resize_width),
resize_height_(resize_height)
{
    model_ = tflite::FlatBufferModel::BuildFromFile(palm_detector_path.c_str());
    tflite::InterpreterBuilder(*model_, resolver_)(&interpreter_);

    interpreter_->AllocateTensors();

    input_tensor_ = interpreter_->typed_input_tensor<float>(0);
    output_tensor_ = interpreter_->typed_output_tensor<float>(0);

}

HandDetector::~HandDetector(){
    // delete [] HandDetector::input_tensor_;
}

void HandDetector::Preprocess(){
    // STEPS
    // - resize orig_img
    // - change color space
    // - regularize 
    // - copy to model input buffer

   cv::Mat resized, resized_rgb, resized_rgb_normalized;

   cv::resize(HandDetector::orig_image, 
             resized, 
             cv::Size(HandDetector::resize_width_, 
             HandDetector::resize_height_), 
             cv::INTER_LINEAR);

    cv::cvtColor(resized, resized_rgb, cv::COLOR_BGR2RGB);
    cv::normalize(resized_rgb, resized_rgb_normalized, -1.0, 1.0, cv::NORM_MINMAX, CV_32F);

    // Copy mat data to input tensor buffer
    std::size_t input_buffer_size = resized_rgb_normalized.total() * resized_rgb_normalized.elemSize();
    // HandDetector::input_tensor_ = new float [input_buffer_size / sizeof(float)];
    memcpy(HandDetector::input_tensor_, resized_rgb_normalized.ptr<float>(0), input_buffer_size);
}

void HandDetector::Inference(){
    interpreter_->Invoke();
}

void HandDetector::Postprocess(){
    // Model output is the following:
    // 1x896x18 tensor of anchors
    // 1x896 tensor of confidence for each anchor

    // To postprocess, let's loop through the confidence tensor and find the index of the highest one
    // Which index to start at? 896*18 to find starting index
    const int num_anchors = 896;
    const int anchor_size = 18;
    const int first_output_size = num_anchors * anchor_size;
    const int total_size = first_output_size + 896;

    float max = 0.0;
    int max_index;
    for(int x=first_output_size; x < total_size; x++){
        if(HandDetector::output_tensor_[x] > max){
            max = HandDetector::output_tensor_[x];
            max_index = x;
        }
    }

    std::cout << "Most confident anchor at index: " << max_index - first_output_size<< std::endl;
    std::cout << "Most confident anchor value   : " << max << std::endl;

    // Print most anchor values of bet index
    const int best_anchor_index = max_index - first_output_size;
    const int best_anchor_index_first_output = best_anchor_index * anchor_size;

    for(int x=0; x < 18; x++){
        std::cout << "Anchor index " << x << " value: " << HandDetector::output_tensor_[x+best_anchor_index_first_output] << std::endl;
    }

    std::vector<cv::Point> points;

    const float scale_width = float(orig_width_) / float(resize_width_);
    const float scale_height = float(orig_height_) / float(resize_height_);

    // for(int x=0; x < 18/2; x+=2){
    //     cv::Point p;
    //     p.x = std::round(HandDetector::output_tensor_[x+best_anchor_index_first_output] * scale_width); // + orig_width_ / 2;
    //     p.y = std::round(HandDetector::output_tensor_[x+best_anchor_index_first_output+1] * scale_height); // + orig_height_ / 2;
    //     points.push_back(p);
    // }

    for(int x=0; x < 18/2; x+=4){
        cv::Point p1;
        cv::Point p2;
        p1.x = std::round(HandDetector::output_tensor_[x+best_anchor_index_first_output] * scale_width); // + orig_width_ / 2;
        p1.y = std::round(HandDetector::output_tensor_[x+best_anchor_index_first_output+1] * scale_height); // + orig_height_ / 2;
        p2.x = p1.x + std::round(HandDetector::output_tensor_[x+best_anchor_index_first_output+2] * scale_width); // + orig_width_ / 2;
        p2.y = p1.y + std::round(HandDetector::output_tensor_[x+best_anchor_index_first_output+3] * scale_height); // + orig_height_ / 2;


        points.push_back(p1);
        points.push_back(p2);
    }
    

    cv::Scalar red( 255, 0, 0 );

    // for(int x=0; x < points.size(); x++){
    //     cv::circle(HandDetector::orig_image, points[x], 7, red, cv::FILLED);
    // }

    for(int x=0; x < points.size(); x+=2){
        cv::rectangle(HandDetector::orig_image, points[x], points[x+1], red);
    }
    // draw on original image to see output

    std::cout << scale_width << std::endl;
    std::cout << scale_height << std::endl;

    cv::imshow("test", HandDetector::orig_image); cv::waitKey(0);
}

/*
void HandDetector::Postprocess(){
    // Model output is the following:
    // 1x896x18 tensor of anchors
    // 1x896 tensor of confidence for each anchor

    // To postprocess, let's loop through the confidence tensor and find the index of the highest one
    // Which index to start at? 896*18 to find starting index
    const int num_anchors = 896;
    const int anchor_size = 18;
    const int first_output_size = num_anchors * anchor_size;
    const int total_size = first_output_size + 896;

    float max = 0.0;
    int max_index;
    for(int x=first_output_size; x < total_size; x++){
        if(HandDetector::output_tensor_[x] > max){
            max = HandDetector::output_tensor_[x];
            max_index = x;
        }
    }

    std::cout << "Most confident anchor at index: " << max_index - first_output_size<< std::endl;
    std::cout << "Most confident anchor value   : " << max << std::endl;

    // Print most anchor values of bet index
    const int best_anchor_index = max_index - first_output_size;
    const int best_anchor_index_first_output = best_anchor_index * anchor_size;

    for(int x=0; x < 18; x++){
        std::cout << "Anchor index " << x << " value: " << HandDetector::output_tensor_[x+best_anchor_index_first_output] << std::endl;
    }

    std::vector<cv::Point> points;

    const float scale_width = float(orig_width_) / float(resize_width_);
    const float scale_height = float(orig_height_) / float(resize_height_);

    // for(int x=0; x < 18/2; x+=2){
    //     cv::Point p;
    //     p.x = std::round(HandDetector::output_tensor_[x+best_anchor_index_first_output] * scale_width); // + orig_width_ / 2;
    //     p.y = std::round(HandDetector::output_tensor_[x+best_anchor_index_first_output+1] * scale_height); // + orig_height_ / 2;
    //     points.push_back(p);
    // }

    for(int x=0; x < 18/2; x+=4){
        cv::Point p1;
        cv::Point p2;
        p1.x = std::round(HandDetector::output_tensor_[x+best_anchor_index_first_output] * scale_width); // + orig_width_ / 2;
        p1.y = std::round(HandDetector::output_tensor_[x+best_anchor_index_first_output+1] * scale_height); // + orig_height_ / 2;
        p2.x = p1.x + std::round(HandDetector::output_tensor_[x+best_anchor_index_first_output+2] * scale_width); // + orig_width_ / 2;
        p2.y = p1.y + std::round(HandDetector::output_tensor_[x+best_anchor_index_first_output+3] * scale_height); // + orig_height_ / 2;


        points.push_back(p1);
        points.push_back(p2);
    }
    

    cv::Scalar red( 255, 0, 0 );

    // for(int x=0; x < points.size(); x++){
    //     cv::circle(HandDetector::orig_image, points[x], 7, red, cv::FILLED);
    // }

    for(int x=0; x < points.size(); x+=2){
        cv::rectangle(HandDetector::orig_image, points[x], points[x+1], red);
    }
    // draw on original image to see output

    std::cout << scale_width << std::endl;
    std::cout << scale_height << std::endl;

    cv::imshow("test", HandDetector::orig_image); cv::waitKey(0);
} */


void HandDetector::Process(cv::Mat orig_image){
    HandDetector::orig_image = orig_image;
    Preprocess();
    Inference();
    Postprocess();
}

