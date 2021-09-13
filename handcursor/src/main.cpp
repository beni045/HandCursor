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
    // Intermediate data
    // Mat orig_image;
    // Mat cropped_frame;
    // output of final coordinate datatype?

    // Modules of pipeline
    // VideoCapture cap;
    // cap.open(10);

    // if(!cap.isOpened())
    //     cout << "Camera failed to open!" << endl;
    //     return 0;
    // cap >> orig_image;  

    std::string image_path = "/home/beni045/Documents/HandCursor_local/HandCursor/handcursor/src/hand_sample.jpg";
    Mat orig_image = imread(image_path, IMREAD_COLOR);
    if(orig_image.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    const unsigned int orig_width = orig_image.cols;
    const unsigned int orig_height = orig_image.rows;

    HandDetector handdetector(orig_width, orig_height, 128, 128);
    // KeypointDetector keypointdetector();

    float* input_tensor = handdetector.Process(orig_image);

    for(int x=0; x < 10; x++){
        cout << input_tensor[x] << endl;
    }

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