#include <cstdio>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main()
{
    // Intermediate data
    Mat orig_frame;
    Mat cropped_frame;
    // output of final coordinate datatype?

    // Modules of pipeline
    VideoCapture cap;
    if(!cap.open(0))
        cout << "Camera failed to open!" << endl;
        return 0;
    cap >> orig_frame;
    const unsigned int orig_width = orig_frame.cols;
    const unsigned int orig_height = orig_frame.rows;

    // HandDetector handdetector(orig_width, orig_height);
    // KeypointDetector keypointdetector();



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