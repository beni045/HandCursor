#include <iostream>
#include <opencv2/highgui.hpp>


class HandDetector{       
    private:            
        const int orig_width, orig_height;  
        const int resize_width, resize_height;  

        cv::Mat orig_image;    

        float* input_tensor;
        float* output_tensor;


    public:
        HandDetector(int orig_width, int orig_height, int resize_width, int resize_height);
        ~HandDetector();
        
        void Preprocess();
        float* Process(cv::Mat orig_image);



        
    // Inference();
    // Postprocess();

    // input: frame
    // output: cropped frame


    // STEPS
    // resize orig_img
    // regularize resized_img
    // format/copy resized_img to model input
    // proc inference
    // postprocess output

};