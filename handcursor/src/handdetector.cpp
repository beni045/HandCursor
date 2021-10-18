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
    last_idx_ = 0;
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

int8_t HandDetector::Postprocess(){
    // Model output is the following:
    // 1x2944x18 tensor of anchors
    // 1x2944 tensor of confidence for each anchor

    const int num_anchors = 2944;
    const int anchor_size = 18;
    const double threshold_val = 0.8;

    /*std::vector<int> threshold_idx;*/
    double max_conf = 0;
    int max_conf_idx = 0;

    // Check if last idx is above threshold
    double last_idx_sigm = sigmoidfunc(double(output_tensor2_[last_idx_]));
    if (last_idx_sigm > threshold_val) {
        max_conf_idx = last_idx_;
        max_conf = last_idx_sigm;
    }

    else {
        for (int x = 0; x < num_anchors; x++) {
            double sigm = sigmoidfunc(double(output_tensor2_[x]));

            // Try using highest conf score
            if (sigm > max_conf) {
                max_conf = sigm;
                max_conf_idx = x;
            }

            //if (sigm > threshold_val){
            //    threshold_idx.push_back(x);
            //}
        }
    }
    //// Check if theres hand
    //if (threshold_idx.empty()){
    //    return NO_DETECT;
    //}

    // implement NMS instead
    /*int widest_box_idx = FindWidest(threshold_idx);*/

    //std::cout << "Max conf idx: " << max_conf_idx << std::endl;

    // Check if theres hand
    if (max_conf < threshold_val) {
        return NO_DETECT;
    }

    // test highest conf idx
    //widest_box_idx = max_conf_idx;

    std::vector<cv::Point> keypoints = FindKeypoints(max_conf_idx);

    int i = 0;
    for(auto kp : keypoints){
        cv::circle(orig_image_, kp, 5, cv::Scalar(255,0,0),cv::FILLED, 8,0);
        //cv::putText(orig_image_,std::to_string(i),kp,cv::FONT_HERSHEY_DUPLEX,1,cv::Scalar(0,255,0),2,false);
        i++;

    }
    cv::Mat transformed = TransformPalm(keypoints[0], keypoints[2], 1);

    result_ = transformed;
    return SUCCESS;
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


std::vector<cv::Point> HandDetector::FindKeypoints(int target_idx){
    const float scale_orig_x = float(orig_width_) / float(resize_width_);
    const float scale_orig_y = float(orig_height_) / float(resize_height_);

    cv::Point center;

    // Find center of anchor box
    center.x = anchors_[target_idx * 2] * 256;
    center.y = anchors_[(target_idx * 2)+1] * 256;

    // Get all 7 keypoints
    std::vector<cv::Point> keypoints;
    // Keypoints are between [target_idx, 4:]
    for(int x = target_idx*18 + 4; x < target_idx*18 + 18; x+=2){
        cv::Point p;
        p.x = (int)std::round((output_tensor1_[x] + center.x) * scale_orig_x);
        p.y = (int)std::round((output_tensor1_[x+1] + center.y) * scale_orig_y);
        keypoints.push_back(p);
    }

    return keypoints;
}


cv::Mat HandDetector::TransformPalm(cv::Point wrist, cv::Point middlefinger, float scale){

    /*std::cout << "wrist: " << wrist << ", middle: " << middlefinger << std::endl;*/

    cv::Mat rotated = cv::Mat::zeros(orig_image_.rows, 
                                         orig_image_.cols, 
                                         orig_image_.type() );
        
    // Set wrist to origin
    cv::Point wrist_to_middle = wrist - middlefinger;
    cv::Point vertical(0, 1);

    // Find angle between wrist to middle and vertical
    // and rotate based on orientation (hand should always point upward)
    float angle = angleBetween(vertical, wrist_to_middle);
    //std::cout << "ANGLE1: " << angle << std::endl;

    if (wrist.x < middlefinger.x){
        angle = angleBetween(vertical, wrist_to_middle);
        //if (angle > 90){
        //    angle = 180 - angle;
        //}
    }
    else {
        angle = -angleBetween(vertical, wrist_to_middle);
        //if (angle < -90){
        //    angle = -180 - angle;
        //}
    }

    //std::cout << "ANGLE2: " << angle << std::endl;


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
    float cropHeight = dist_wrist_middle * 3.5 * scale;
    float cropWidth = dist_wrist_middle * 4 * scale;
    float x_offset = (wrist.x - (cropWidth / 2.0)) * scale;
    //float x_offset = (2 * wrist.x) - cropWidth;
    float y_offset = (wrist.y - (cropHeight * 1)) * 1 * scale; // make sure theres some space below wrist

    // Limit crop size
    if (x_offset > rotated.cols){
        x_offset = rotated.cols - 1;
    }
    else if(x_offset < 0){
        x_offset = 1;
    }
    if (y_offset > rotated.rows){
        y_offset = rotated.rows - 1;
    }
    else if(y_offset < 0){
        y_offset = 1;
    }
    if ((x_offset + cropWidth) > rotated.cols){
        cropWidth = rotated.cols - x_offset - 1;
    }
    if (cropWidth <= 0) {
        cropWidth = 1;
    }
    if ((y_offset + cropHeight) > rotated.rows){
        cropHeight = rotated.rows - y_offset - 1;
    }
    if (cropHeight <= 0) {
        cropHeight = 1;
    }

    // Save offsets for later transformback of keypoints
    transdata_.offset = cv::Point2f(x_offset, y_offset);

    
    cv::Rect ROI(x_offset, y_offset, cropWidth, cropHeight);

    //std::cout << "Crop rect, angle: " << ROI << "  " << angle << std::endl;

    cv::Mat croppedImage = rotated(ROI);   

    // Save rotated rect for display
    cropRect_ = cv::RotatedRect(middlefinger, cv::Size(cropWidth, cropHeight*1.1), angle);

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

void HandDetector::PostprocessExternalKps(cv:: Point2f wrist, cv::Point2f middlefinger){
    cv::Mat transformed = TransformPalm(wrist, middlefinger, 1);
    std::cout << "^ 2nd type" << std::endl;
    result_ = transformed;
}

void HandDetector::ReadFrame(cv::Mat frame) {
    orig_image_ = frame;
}

void HandDetector::TransformPalm2(std::vector<cv::Point2f> keypoints, float scale) {
    // rotate orig frame
    // rotate keypoints
    // crrop based on rectangle
        /*std::cout << "wrist: " << wrist << ", middle: " << middlefinger << std::endl;*/

    cv::Mat rotated = cv::Mat::zeros(orig_image_.rows,
        orig_image_.cols,
        orig_image_.type());

    // Set wrist to origin
    cv::Point wrist = keypoints[0];
    cv::Point middlefinger = keypoints[9];
    cv::Point wrist_to_middle = wrist - middlefinger;
    cv::Point vertical(0, 1);

    // Find angle between wrist to middle and vertical
    // and rotate based on orientation (hand should always point upward)
    float angle = angleBetween(vertical, wrist_to_middle);
    //std::cout << "ANGLE1: " << angle << std::endl;

    if (wrist.x < middlefinger.x) {
        angle = angleBetween(vertical, wrist_to_middle);
        //if (angle > 90){
        //    angle = 180 - angle;
        //}
    }
    else {
        angle = -angleBetween(vertical, wrist_to_middle);
        //if (angle < -90){
        //    angle = -180 - angle;
        //}
    }

    //std::cout << "ANGLE2: " << angle << std::endl;


    // Save wrist as center of rotation and angle for transformback of kps
    transdata_.center = wrist;
    transdata_.angleRad = (double(angle) / RAD_TO_DEG);

    //Apply rotation transform
    cv::Mat rot_mat = cv::getRotationMatrix2D(wrist, angle, 1);   // TODO: test this scale
    cv::warpAffine(orig_image_, rotated, rot_mat, rotated.size());


    // Also rotate points:
    for (auto& p : keypoints) {
        p = rotatePoint(p, wrist, -angle / RAD_TO_DEG);
    }

    // Find min and max points
    cv::Point2f max = cv::Point2f(0, 0);
    cv::Point2f min = cv::Point2f(orig_image_.cols, orig_image_.rows);

    for (const auto p : keypoints) {
        if (p.x > max.x) {
            max.x = p.x;
        }
        if (p.y > max.y) {
            max.y = p.y;
        }
        if (p.x < min.x) {
            min.x = p.x;
        }
        if (p.y < min.y) {
            min.y = p.y;
        }
    }

    // Scale bounding box
    min /= scale;
    max *= scale;


    // Prevent error-feedback when cropping
    int bbox_width = max.x - min.x;
    int bbox_height = max.y - min.y;
    if (bbox_width > bbox_height) {
        float diff = (bbox_width - bbox_height) / 2;
        min.y -= diff;
        max.y += diff;
    }
    else  {
        float diff = (bbox_height - bbox_width) / 2;
        min.x -= diff;
        max.x += diff;
    }


    // Limit crop dims to image size
    if (max.x > rotated.cols) {
        max.x = rotated.cols;
    }

    if (max.y > rotated.rows) {
        max.y = rotated.rows;
    }

    if (min.x < 0) {
        min.x = 0;
    }

    if (min.y < 0) {
        min.y = 0;
    }

    
    cv::Rect ROI = cv::Rect(min, max);
    cv::Mat cropped_image = rotated(ROI);

    transdata_.offset = min;

    result_ = cropped_image;
    /*cv::imwrite("cropped.png", result_);*/

    return;
}