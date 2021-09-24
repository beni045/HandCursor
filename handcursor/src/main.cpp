#include <cstdio>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <handdetector.h>


using namespace std;
using namespace cv;

int main()
{
    string image_path = "/home/beni045/Documents/HandCursor_local/HandCursor/handcursor/src/sample_hand3.jpg";
    string palm_detector_path = "/home/beni045/Documents/HandCursor_local/HandCursor/models/palm_detection_without_custom_op.tflite";
    const int resize_width = 256;
    const int resize_height = 256;


    Mat orig_image = imread(image_path, IMREAD_COLOR);
    if(orig_image.empty())
    {
        cout << "Could not read the image: " << image_path << endl;
        return 1;
    }

    const unsigned int orig_width = orig_image.cols;
    const unsigned int orig_height = orig_image.rows;

    HandDetector handdetector(orig_width, orig_height, resize_width, resize_height, palm_detector_path);
    
    handdetector.Process(orig_image);

    // cout << handdetector.orig_width << endl;

    // if(!cap.open(0))
    //     cout << "cap not open.." << endl;
    //     return 0;
    // for(;;)
    // {
    //       Mat frame;
    //       cap >> frame;
    //       if( frame.empty() ) break; // end of video stream
    //       imshow("this is you, smile! :)", frame);
    //       if( waitKey(10) == 27 ) break; // stop capturing by pressing ESC 
    // }
    // return 0;
    return 0;
}