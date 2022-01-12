#include <cstdio>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
//#include <opencv2/highgui.hpp>
//#include <opencv2/core.hpp>
//#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <handdetector.h>
#include <keypointdetector.h>
#include <utils.h>
#include <chrono>
#include <thread>

using namespace std;
using namespace cv;

int main()
{
    string image_path = "../../src/sample_hand3.jpg";
    string palm_detector_path = "../../../models/palm_detection_without_custom_op.tflite";
    string keypoint_detector_path = "../../../models/hand_landmark_new.tflite";
    
    const int resize_width = 256;
    const int resize_height = 256;

    Mat orig_image = imread(image_path, IMREAD_COLOR);
    if(orig_image.empty())
    {
        cout << "Could not read the image: " << image_path << endl;
        return 1;
    }

    HandDetector handdetector(resize_width, resize_height, palm_detector_path);
    KeypointDetector keypointdetector(224, 224, keypoint_detector_path);

    //Mat cropped_img;
    //handdetector.Process(orig_image);
    //cropped_img = handdetector.GetResult();
    //keypointdetector.Process(cropped_img);

    // imshow("Cropped Image", cropped_img);
    // waitKey(0);


    VideoCapture cap;

    if (!cap.open(0)) {
        cout << "cap not open.." << endl;
        return 0;
    }

    Mat frame;
    Mat flipped;
    Mat cropped_img;
    vector<Point2f> final_kps;
    int8_t status;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    bool FIND_PALM = true;

     for(;;)
     {
           //std::this_thread::sleep_for(std::chrono::milliseconds(1000));
           cap >> frame;
               
           if( frame.empty() ) break; // end of video stream

           if (FIND_PALM) {
               // Handle no palm detected
               begin = chrono::steady_clock::now();
               status = handdetector.Process(frame);
               end = chrono::steady_clock::now();
               //cout << "palm time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << endl;
               if (status == NO_DETECT) {

                   flip(frame, flipped, 1);
                   imshow("Keypoint Overlay", flipped);
                   if (waitKey(10) == 27) break;
                   cout << "no palm" << endl;
                   continue;
               }
           }

           else {
               handdetector.ReadFrame(frame);
               //handdetector.PostprocessExternalKps(final_kps[0], final_kps[9]);
               handdetector.TransformPalm2(final_kps, 1.25);
               /*cropped_img = handdetector.GetResult();*/
               // imshow("Keypoint Overlay", frame);
               if (waitKey(10) == 27) break;
           }
           

           cropped_img = handdetector.GetResult();



           // Handle no hand detected
           begin = chrono::steady_clock::now();
           status = keypointdetector.Process(cropped_img);
           end = chrono::steady_clock::now();
           //cout << "hand time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << endl;
           if (status == NO_DETECT){
               handdetector.ResetPredictor();
               flip(frame, flipped, 1);
               imshow("Keypoint Overlay", flipped);
               if (waitKey(10) == 27) break;

               cout << "no hand" << endl;
               FIND_PALM = true;
               continue;
           }
           else {
               FIND_PALM = false;
           }
           final_kps = keypointdetector.GetResult();
           //cout << "\n orig: " << endl;
           //for (auto p : final_kps) {
           //    cout << "Kps: " << p << endl;
           //}
           //cout << "\n transormed: " << endl;
           handdetector.TransformBack(final_kps);
           for (auto& p : final_kps) {
               int circle_size = int(float(frame.cols) * 0.01);
               cv::circle(frame, p, circle_size, Scalar(0, 255, 0), cv::FILLED, circle_size, 0);
               //cout << "Kps: " << p << endl;
           }
           cv::line(frame, final_kps[0], final_kps[9], cv::Scalar(0, 0, 255));
           cv::Point2f vertices[4];
           handdetector.GetCropRect().points(vertices);
           //for (int i = 0; i < 4; i++) {
           //    cv::line(frame, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0));
           //}
           // cv::rectangle(frame, handdetector.GetCropRect().(), cv::Scalar(0, 255, 0));
           

           flip(frame, flipped, 1);
           imshow("Keypoint Overlay", flipped);
           if (waitKey(10) == 27) break; // stop capturing by pressing ESC 
     }
    // return 0;
    return 0;
}