#include <iostream>
#include <opencv2/highgui.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "modelprocessor.h"


class KeypointDetector : public ModelProcessor {       
    private:            
        float* output_tensor3_;

        void Preprocess();
        void Postprocess();

        int FindWidest(std::vector<int> threshold_idxs);

        std::vector<cv::Point> FindKeypoints(int widest_idx);
        cv::Mat TransformPalm(cv::Point wrist, cv::Point middlefinger, float thirdpoint_scale);

    public:
        KeypointDetector(int resize_width, int resize_height, std::string model_path);
};