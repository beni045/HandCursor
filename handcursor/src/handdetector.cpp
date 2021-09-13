#include <handdetector.h>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

HandDetector::HandDetector(int orig_width, int orig_height, int resize_width, int resize_height)
:
orig_width(orig_width),
orig_height(orig_height),
resize_width(resize_width),
resize_height(resize_height)
{
    // HandDetector::orig_width = orig_width;
}

HandDetector::~HandDetector(){
    delete [] HandDetector::input_tensor;
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
             cv::Size(HandDetector::resize_width, 
             HandDetector::resize_height), 
             cv::INTER_LINEAR);

    cv::cvtColor(resized, resized_rgb, cv::COLOR_BGR2RGB);
    cv::normalize(resized_rgb, resized_rgb_normalized, -1, 1, cv::NORM_MINMAX, CV_32F);

    // std::cout << "input data size: " << resized_rgb_normalized.size() << std::endl;

    // std::cout << "input data size (bytes): " << resized_rgb_normalized.total() * resized_rgb_normalized.elemSize() << std::endl;


    // Copy mat data to input tensor buffer
    std::size_t input_buffer_size = resized_rgb_normalized.total() * resized_rgb_normalized.elemSize();
    HandDetector::input_tensor = new float [input_buffer_size / sizeof(float)];
    memcpy(HandDetector::input_tensor, resized_rgb_normalized.ptr<float>(0), input_buffer_size);
}

float* HandDetector::Process(cv::Mat orig_image){
    HandDetector::orig_image = orig_image;
    Preprocess();
    return HandDetector::input_tensor;
}