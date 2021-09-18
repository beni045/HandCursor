#include <handdetector.h>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include <iostream>
#include <fstream>
#include <utils.h>
#include <cstdio>

#define ANCHORS_PATH "/home/beni045/Documents/HandCursor_local/HandCursor/models/anchors.csv"


#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }



HandDetector::HandDetector(int orig_width, int orig_height, int resize_width, int resize_height, std::string palm_detector_path)
:
orig_width_(orig_width),
orig_height_(orig_height),
resize_width_(resize_width),
resize_height_(resize_height)
{
    model_ = tflite::FlatBufferModel::BuildFromFile(palm_detector_path.c_str());
    TFLITE_MINIMAL_CHECK(model_ != nullptr);

    tflite::InterpreterBuilder(*model_, resolver_)(&interpreter_);

    TFLITE_MINIMAL_CHECK(interpreter_ != nullptr);
    TFLITE_MINIMAL_CHECK(interpreter_->AllocateTensors() == kTfLiteOk);

    input_tensor_ = interpreter_->typed_input_tensor<float>(0);
    output_tensor1_ = interpreter_->typed_output_tensor<float>(0);
    output_tensor2_ = interpreter_->typed_output_tensor<float>(1);

    // tflite::PrintInterpreterState(interpreter_.get());

    anchors_ = LoadAnchors(ANCHORS_PATH);
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
    // - 

   cv::Mat resized, resized_rgb, resized_rgb_normalized;

   cv::resize(HandDetector::orig_image_, 
             resized, 
             cv::Size(HandDetector::resize_width_, 
             HandDetector::resize_height_), 
             cv::INTER_LINEAR);

    cv::cvtColor(resized, resized_rgb, cv::COLOR_BGR2RGB);
    cv::normalize(resized_rgb, resized_rgb_normalized, -1.0, 1.0, cv::NORM_MINMAX, CV_32F);

    // Copy mat data to input tensor buffer
    std::size_t input_buffer_size = resized_rgb_normalized.total() * resized_rgb_normalized.elemSize();

    float* preproc_mat = resized_rgb_normalized.ptr<float>(0);
    
    memcpy(HandDetector::input_tensor_, preproc_mat, input_buffer_size);
}

void HandDetector::Inference(){
    TFLITE_MINIMAL_CHECK(interpreter_->Invoke() == kTfLiteOk);
}

void HandDetector::Postprocess(){
    // Model output is the following:
    // 1x2944x18 tensor of anchors
    // 1x2944 tensor of confidence for each anchor

    const int num_anchors = 2944;
    const int anchor_size = 18;
    const double threshold_val = 0.7;

    std::vector<int> threshold_idx;


    for (int x=0; x < num_anchors; x++){
        double sigm = sigmoidfunc(double(output_tensor2_[x]));
        if (sigm > threshold_val){
            threshold_idx.push_back(x);
        }
        //std::cout << "x: " << x - first_output_size << "  sigm: " << sigm << std::endl;
    }
    std::cout << "threshold idx size: " << threshold_idx.size() << std::endl;

    // for(int x= 0; x < 5; x++){
    //     std::cout << "output cls: " << output_tensor2_[x] << std::endl;
    // }

    // for(int x= 0; x < 10; x++){
    //     std::cout << "input tensor : " << input_tensor_[x] << std::endl;
    // }
  

    // IF NO IDX ABOVE THRESHOLD (SIZE = 0), RETURN NO HAND!


    // find argmax for widest box in candidate detect (dim 3)

    int widest_box_idx = FindWidest(threshold_idx);

    std::cout << "widest box idx: " << widest_box_idx << std::endl;


    // find anchor for that index

    cv::Rect bbox = FindBbox(widest_box_idx);
    DrawBboxOrig(bbox);
    
    cv::Mat resize_show;

    cv::resize(HandDetector::orig_image_, 
             resize_show, 
             cv::Size(500, 500), 
             cv::INTER_LINEAR);

    cv::imshow("test", resize_show); cv::waitKey(0);

    // find bounding box based on anchor and bbox from output




    // # bounding box offsets, width and height
    //     dx,dy,w,h = candidate_detect[max_idx, :4]
    //     center_wo_offst = candidate_anchors[max_idx,:2] * 256

        // detecion_mask = self._sigm(out_clf) > 0.7
        // candidate_detect = out_reg[detecion_mask]
        // candidate_anchors = self.anchors[detecion_mask]

        // if candidate_detect.shape[0] == 0:
        //     print("No hands found")
        //     return None, None, None

    // sigmoid function out output classifiers
    // find elements above threshold (indices)
    // use those indices to get detections
    // use those idices to get anchors
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
        if(HandDetector::output_tensor1_[x] > max){
            max = HandDetector::output_tensor1_[x];
            max_index = x;
        }
    }

    std::cout << "Most confident anchor at index: " << max_index - first_output_size<< std::endl;
    std::cout << "Most confident anchor value   : " << max << std::endl;

    // Print most anchor values of bet index
    const int best_anchor_index = max_index - first_output_size;
    const int best_anchor_index_first_output = best_anchor_index * anchor_size;

    for(int x=0; x < 18; x++){
        std::cout << "Anchor index " << x << " value: " << HandDetector::output_tensor1_[x+best_anchor_index_first_output] << std::endl;
    }

    std::vector<cv::Point> points;

    const float scale_width = float(orig_width_) / float(resize_width_);
    const float scale_height = float(orig_height_) / float(resize_height_);

    // for(int x=0; x < 18/2; x+=2){
    //     cv::Point p;
    //     p.x = std::round(HandDetector::output_tensor1_[x+best_anchor_index_first_output] * scale_width); // + orig_width_ / 2;
    //     p.y = std::round(HandDetector::output_tensor1_[x+best_anchor_index_first_output+1] * scale_height); // + orig_height_ / 2;
    //     points.push_back(p);
    // }

    for(int x=0; x < 18/2; x+=4){
        cv::Point p1;
        cv::Point p2;
        p1.x = std::round(HandDetector::output_tensor1_[x+best_anchor_index_first_output] * scale_width); // + orig_width_ / 2;
        p1.y = std::round(HandDetector::output_tensor1_[x+best_anchor_index_first_output+1] * scale_height); // + orig_height_ / 2;
        p2.x = p1.x + std::round(HandDetector::output_tensor1_[x+best_anchor_index_first_output+2] * scale_width); // + orig_width_ / 2;
        p2.y = p1.y + std::round(HandDetector::output_tensor1_[x+best_anchor_index_first_output+3] * scale_height); // + orig_height_ / 2;


        points.push_back(p1);
        points.push_back(p2);
    }
    

    cv::Scalar red( 255, 0, 0 );

    // for(int x=0; x < points.size(); x++){
    //     cv::circle(HandDetector::orig_image_, points[x], 7, red, cv::FILLED);
    // }

    for(int x=0; x < points.size(); x+=2){
        cv::rectangle(HandDetector::orig_image_, points[x], points[x+1], red);
    }
    // draw on original image to see output

    std::cout << scale_width << std::endl;
    std::cout << scale_height << std::endl;

    cv::imshow("test", HandDetector::orig_image_); cv::waitKey(0);
} */


int HandDetector::FindWidest(std::vector<int> thresh_idxs){
    float max = 0;
    int max_idx = 0;
    for(int i : thresh_idxs){
        std::cout << "idx: " << i << "  width: " << output_tensor1_[(i*18)+3] << std::endl;

        if (output_tensor1_[(i*18)+3] > max){
            max = output_tensor1_[(i*18)+3];
            max_idx = i;
        }
    }
    return max_idx;
}


cv::Rect HandDetector::FindBbox(int widest_idx){
    const float scale_orig_x = orig_width_ / resize_width_;
    const float scale_orig_y = orig_height_ / resize_height_;

    cv::Point center;

    // Find center of anchor box
    center.x = anchors_[widest_idx * 2] * 256;
    center.y = anchors_[(widest_idx * 2)+1] * 256;

    // Find shifted center based on model output
    center.x += output_tensor1_[widest_idx*18];
    center.y += output_tensor1_[(widest_idx*18)+1];

    cv::Point topLeft;
    cv::Point bottomRight;

    // Shift center based on width,height of box to find points
    // Also scale to orig image coordinates and convert to int
    float shift_x = output_tensor1_[((widest_idx*18)+2)/2];
    float shift_y = output_tensor1_[((widest_idx*18)+3)/2];
    topLeft.x = int((center.x + shift_x) * scale_orig_x);
    topLeft.y = int((center.y + shift_y) * scale_orig_y);
    bottomRight.x = int((center.x - shift_x) * scale_orig_x);
    bottomRight.x = int((center.x - shift_y) * scale_orig_y);

    cv::Rect bbox(topLeft, bottomRight);
    return bbox;
}

void HandDetector::DrawBboxOrig(cv::Rect bbox){
    cv::Scalar red( 0, 0, 255);
    cv::rectangle(orig_image_, bbox, red);
}


void HandDetector::Process(cv::Mat orig_image_){
    HandDetector::orig_image_ = orig_image_;
    Preprocess();
    Inference();
    Postprocess();
}




std::vector<float> HandDetector::LoadAnchors(std::string filepath){
    std::vector<float> anchors;

    std::ifstream myFile;
    myFile.open(filepath);

    int idx = 0;
    while (myFile.good()){
        std::string line;
        getline(myFile, line, ',');
        anchors.push_back(std::atof(line.c_str()));
        getline(myFile, line, '\n');
        anchors.push_back(std::atof(line.c_str()));
    }

    return anchors;
}