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

using namespace std;
using namespace cv;

int main()
{
    string image_path = "C:/Users/benig/Documents/Projects/Hand_controls_cpp/windows_version/tflite/handcursor/src/sample_hand3.jpg";
    string palm_detector_path = "C:/Users/benig/Documents/Projects/Hand_controls_cpp/windows_version/tflite/models/palm_detection_without_custom_op.tflite";
    string keypoint_detector_path = "C:/Users/benig/Documents/Projects/Hand_controls_cpp/windows_version/tflite/models/hand_landmark_new.tflite";
    
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
    Mat cropped_img;
    vector<Point2f> final_kps;
     for(;;)
     {
           cap >> frame;
           if( frame.empty() ) break; // end of video stream

           handdetector.Process(frame);
           cropped_img = handdetector.GetResult();
           keypointdetector.Process(cropped_img);
           final_kps = keypointdetector.GetResult();
           //cout << "\n orig: " << endl;
           //for (auto p : final_kps) {
           //    cout << "Kps: " << p << endl;
           //}
           //cout << "\n transormed: " << endl;
           //handdetector.TransformBack(final_kps);
           for (auto p : final_kps) {
               cv::circle(frame, p, 10, Scalar(0, 255, 0), cv::FILLED, 8, 0);
               //cout << "Kps: " << p << endl;
           }

           imshow("Keypoint Overlay", frame);
           if (waitKey(10) == 27) break; // stop capturing by pressing ESC 
     }
    // return 0;
    return 0;
}