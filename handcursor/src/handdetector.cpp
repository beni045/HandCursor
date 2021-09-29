#include <handdetector.h>
//#include <opencv2/highgui.hpp>
//#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include <iostream>
#include <fstream>
#include <utils.h>
#include <cstdio>

#define ANCHORS_PATH "../../../models/anchors.csv"


HandDetector::HandDetector(int resize_width, 
                           int resize_height, 
                           std::string model_path)
                           :ModelProcessor(resize_width, 
                                           resize_height,
                                           model_path)
{
    ExtraSetup();
}


void HandDetector::ExtraSetup(){
    anchors_ = LoadAnchors(ANCHORS_PATH);
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
    }
    
    int widest_box_idx = FindWidest(threshold_idx);

    std::vector<cv::Point> keypoints = FindKeypoints(widest_box_idx);

    int i = 0;
    for(auto kp : keypoints){
        //cv::circle(orig_image_, kp, 10, cv::Scalar(0,255,0),cv::FILLED, 8,0);
        //cv::putText(orig_image_,std::to_string(i),kp,cv::FONT_HERSHEY_DUPLEX,1,cv::Scalar(0,255,0),2,false);
        i++;

    }

    cv::Mat transformed = TransformPalm(keypoints[0], keypoints[2], 0.7);

    result_ = transformed;
    // result_ = orig_image_;
}



int HandDetector::FindWidest(std::vector<int> thresh_idxs){
    float max = 0;
    int max_idx = 0;
    for(int i : thresh_idxs){
        if (output_tensor1_[(i*18)+3] > max){
            max = output_tensor1_[(i*18)+3];
            max_idx = i;
        }
    }
    return max_idx;
}


std::vector<cv::Point> HandDetector::FindKeypoints(int widest_idx){
    const float scale_orig_x = float(orig_width_) / float(resize_width_);
    const float scale_orig_y = float(orig_height_) / float(resize_height_);

    cv::Point center;

    // Find center of anchor box
    center.x = anchors_[widest_idx * 2] * 256;
    center.y = anchors_[(widest_idx * 2)+1] * 256;

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
    cv::Point vertical(0, 1);

    // Find angle between wrist to middle and vertical
    // and rotate based on orientation (hand should always point upward)
    float angle = angleBetween(vertical, wrist_to_middle);
    if (wrist.x < middlefinger.x){
        angle = angleBetween(vertical, wrist_to_middle);
        if (angle > 90){
            angle = 180 - angle;
        }
    }
    else {
        angle = -angleBetween(vertical, wrist_to_middle);
        if (angle < -90){
            angle = -180 - angle;
        }
    }

    // Save wrist as center of rotation and angle for transformback of kps
    transdata_.center = wrist;
    transdata_.angleRad = (double(angle) / RAD_TO_DEG);

    //Apply rotation transform
    cv::Mat rot_mat = cv::getRotationMatrix2D(wrist, angle, 1);   // TODO: test this scale
    cv::warpAffine(orig_image_, rotated, rot_mat, rotated.size());


    /* ---Crop based on wrist--- 
    ------------------------------*/
    float dist_wrist_middle = cv::norm(wrist - middlefinger);

    // Heuristic crop params scaled from distance between
    // wrist and middle finger
    float cropHeight = dist_wrist_middle * 2.5;
    float cropWidth = dist_wrist_middle * 2.5;
    float x_offset = wrist.x - cropWidth / 2;
    float y_offset = (wrist.y - cropHeight * 0.85) * 0.9; // make sure theres some space below wrist

    // Save offsets for later transformback of keypoints
    transdata_.offset = cv::Point2f(x_offset, y_offset);

    // Limit crop size
    if (x_offset > orig_image_.cols){
        x_offset = orig_image_.cols - 1;
    }
    else if(x_offset < 0){
        x_offset = 1;
    }
    if (y_offset > orig_image_.rows){
        y_offset = orig_image_.rows - 1;
    }
    else if(y_offset < 0){
        y_offset = 1;
    }
    if ((x_offset + cropWidth) > orig_image_.cols){
        cropWidth = orig_image_.cols - x_offset - 1;
    }
    if ((y_offset + cropHeight) > orig_image_.rows){
        cropHeight = orig_image_.rows - y_offset - 1;
    }
    cv::Rect ROI(x_offset, y_offset, cropWidth, cropHeight);
    cv::Mat croppedImage = rotated(ROI);   

    // Save rotated rect for display
    cv::Point2f rect_center;
    rect_center.x = x_offset + (cropWidth / 2);
    rect_center.y = y_offset + (cropHeight / 2);
    cropRect_ = cv::RotatedRect(rect_center, cv::Size(cropWidth, cropHeight), angle);

    return croppedImage;
    //return rotated;
}

void HandDetector::TransformBack(std::vector<cv::Point2f>& inPoints) {
    // Translate points by crop offset
    // Rotate points back based on angle and wrist as center (same as before)
    for (auto& p : inPoints) {
        p += HandDetector::transdata_.offset;
        p = rotatePoint(p, transdata_.center, transdata_.angleRad);
    }
}

std::vector<float> HandDetector::LoadAnchors(std::string filepath){
    std::vector<float> anchors;

    std::ifstream myFile;
    myFile.open(filepath);
    
    // TODO: refactor error handling
    if(!myFile){
        std::cout << "fail to open anchor file" << std::endl;
    }

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

cv::Mat HandDetector::GetResult(){
    return result_;
}

cv::RotatedRect HandDetector::GetCropRect(){
    return cropRect_;
}