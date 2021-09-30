#include <keypointdetector.h>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include <iostream>
#include <fstream>
#include <cstdio>

KeypointDetector::KeypointDetector(
                           int resize_width, 
                           int resize_height, 
                           std::string model_path)
                           :ModelProcessor(resize_width, 
                                           resize_height,
                                           model_path)
{
    output_tensor3_ = interpreter_->typed_output_tensor<float>(2);

}


void KeypointDetector::Preprocess(){
   cv::Mat resized, resized_rgb, resized_rgb_normalized;

   cv::resize(KeypointDetector::orig_image_, 
             resized, 
             cv::Size(KeypointDetector::resize_width_, 
             KeypointDetector::resize_height_), 
             cv::INTER_LINEAR);

    cv::cvtColor(resized, resized_rgb, cv::COLOR_BGR2RGB);
    cv::normalize(resized_rgb, resized_rgb_normalized, 0.0, 1.0, cv::NORM_MINMAX, CV_32F);

    // Copy mat data to input tensor buffer
    std::size_t input_buffer_size = resized_rgb_normalized.total() * resized_rgb_normalized.elemSize();

    float* preproc_mat = resized_rgb_normalized.ptr<float>(0);

    // cv::imshow("preproc", resized_rgb_normalized);
    // cv::waitKey(0);
    
    memcpy(KeypointDetector::input_tensor_, preproc_mat, input_buffer_size);
}


int8_t KeypointDetector::Postprocess(){
    // std::cout << "Hand presence: " << *output_tensor2_ << std::endl;
    // std::cout << "Handedness: " << *output_tensor3_ << std::endl;

    // Check if hand present
    const float hand_presence_thresh = 0.9;
    if (*output_tensor2_ < hand_presence_thresh){
        return NO_DETECT;
    }

    const float scale_x = float(orig_width_ )/ float(resize_width_);
    const float scale_y = float(orig_height_) / float(resize_height_);

    // Load keypoints from model output
    std::vector<cv::Point2f> keypoints;
    for(int x=0; x < 63; x+=3){
        cv::Point2f kp;
        kp.x = (int)std::round((output_tensor1_[x] * scale_x));
        kp.y = (int)std::round((output_tensor1_[x+1] * scale_y));
        // std::cout << kp;
        // cv::circle(orig_image_, kp, 5, cv::Scalar(0,255,0),cv::FILLED, 8,0);
        keypoints.push_back(kp);
    }


    
    result_ = keypoints;
    return SUCCESS;
}

std::vector<cv::Point2f> KeypointDetector::GetResult(){
    return result_;
}

