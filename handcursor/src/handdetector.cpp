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


    // for(int x= 0; x < 5; x++){
    //     std::cout << "output cls: " << output_tensor2_[x] << std::endl;
    // }

    // for(int x= 0; x < 10; x++){
    //     std::cout << "input tensor : " << input_tensor_[x] << std::endl;
    // }
  

    // IF NO IDX ABOVE THRESHOLD (SIZE = 0), RETURN NO HAND!


    // find argmax for widest box in candidate detect (dim 3)

    int widest_box_idx = FindWidest(threshold_idx);



    // find anchor for that index

    // cv::Rect bbox = FindBbox(widest_box_idx);
    // DrawBboxOrig(bbox);

    std::vector<cv::Point> keypoints = FindKeypoints(widest_box_idx);

    int i = 0;
    for(auto kp : keypoints){
        cv::circle(orig_image_, kp, 10, cv::Scalar(0,255,0),cv::FILLED, 8,0);
        cv::putText(orig_image_,std::to_string(i),kp,cv::FONT_HERSHEY_DUPLEX,1,cv::Scalar(0,255,0),2,false);
        i++;

    }

    cv::Mat transformed = TransformPalm(keypoints[0], keypoints[2], 0.7);


    cv::imshow("transformed", transformed); cv::waitKey(0);

    // cv::resize(HandDetector::orig_image_, 
    //          resize_show, 
    //          cv::Size(500, 500), 
    //          cv::INTER_LINEAR);

    // cv::imshow("test", resize_show); cv::waitKey(0);

    // cv::imshow("orig", orig_image_); cv::waitKey(0);

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


std::vector<cv::Point> HandDetector::FindKeypoints(int widest_idx){
    const float scale_orig_x = orig_width_ / resize_width_;
    const float scale_orig_y = orig_height_ / resize_height_;

    cv::Point center;

    // Find center of anchor box
    center.x = anchors_[widest_idx * 2] * 256;
    center.y = anchors_[(widest_idx * 2)+1] * 256;

    std::cout << "Center anchors: " << center << std::endl;

    // Get all 7 keypoints
    std::vector<cv::Point> keypoints;
    // Keypoints are between [widest_idx, 4:]
    for(int x = widest_idx*18 + 4; x < widest_idx*18 + 18; x+=2){
        cv::Point p;
        p.x = (output_tensor1_[x] + center.x) * scale_orig_x;
        p.y = (output_tensor1_[x+1] + center.y) * scale_orig_y;
        keypoints.push_back(p);
    }

    return keypoints;
}

cv::Mat HandDetector::TransformPalm(cv::Point wrist, cv::Point middlefinger, float thirdpoint_scale){
    cv::Mat rotated = cv::Mat::zeros(orig_image_.rows, 
                                         orig_image_.cols, 
                                         orig_image_.type() );
        
    // Set wrist to origin
    cv::Point wrist_to_middle = middlefinger - wrist;
    cv::Point vertical(0,1);

    // Find angle between wrist to middle and vertical
    float angle = -angleBetween(vertical, wrist_to_middle);
    float scale = 1;

    //Apply rotation transform
    cv::Mat rot_mat = cv::getRotationMatrix2D(wrist, angle, scale);
    cv::warpAffine(orig_image_, rotated, rot_mat, rotated.size());


    // Crop based on wrist
    float dist_wrist_middle = cv::norm(wrist - middlefinger);

    float cropHeight = dist_wrist_middle * 2;
    float cropWidth = dist_wrist_middle * 2.2;

    cv::Rect myROI(wrist.x - cropWidth/2, wrist.y - cropHeight*0.85, cropWidth, cropHeight);
    cv::Mat croppedImage = rotated(myROI);   

    return croppedImage;
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
    float shift_x = output_tensor1_[(widest_idx*18)+2] / 2;
    float shift_y = output_tensor1_[(widest_idx*18)+3] / 2;
    topLeft.x = int((center.x + shift_x) * scale_orig_x);
    topLeft.y = int((center.y + shift_y) * scale_orig_y);
    bottomRight.x = int((center.x - shift_x) * scale_orig_x);
    bottomRight.x = int((center.x - shift_y) * scale_orig_y);

    cv::Rect bbox(topLeft, bottomRight);
    return bbox;
}


void HandDetector::DrawBboxOrig(cv::Rect bbox){
    cv::Scalar green( 0, 255, 0);
    cv::rectangle(orig_image_, bbox, green, 5);
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